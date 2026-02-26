import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import re
import unicodedata
from langchain_core.documents import Document
import json

# Constants
PDF_PATH = "egyptian_labor_law.pdf"
DB_FAISS_PATH = "vectorstore/db_faiss_v3" 
# Set up Page Layout
st.set_page_config(page_title="مساعد قانون العمل المصري", layout="centered")

# Custom CSS for Arabic support and styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri&family=Cairo:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }
    .stChatMessage {
        direction: rtl;
        text-align: right;
    }
    .stTextInput {
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ مساعد قانون العمل المصري")
st.markdown("---")

# Initialize LLM
if "GOOGLE_API_KEY" not in os.environ:
    st.error("الرجاء ضبط مفتاح API الخاص بـ Gemini في متغيرات البيئة (GOOGLE_API_KEY).")
    st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Initialize Embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

def eastern_to_western(text):
    eastern_digits = '٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹'
    western_digits = '01234567890123456789'
    map_table = str.maketrans(eastern_digits, western_digits)
    return text.translate(map_table)

def normalize_arabic_text(text):
    if not text:
        return ""
    # Explicitly convert digits first
    text = eastern_to_western(text)
    # NFKC normalizes presentation forms (ligatures) to standard Arabic characters.
    return unicodedata.normalize('NFKC', text)

# Document Processing and Vector Store Creation
@st.cache_resource
def get_vector_store():
    embeddings = get_embeddings()
    
    if os.path.exists(DB_FAISS_PATH):
        try:
            return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            # If load fails, we recreate
            pass
    
    if not os.path.exists(PDF_PATH):
        st.error(f"ملف القانون غير موجود: {PDF_PATH}")
        st.stop()
        
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    
    # Article-level splitting logic with normalization
    full_text = ""
    page_map = [] 
    
    current_pos = 0
    for page in pages:
        # Normalize text at extraction time
        text = normalize_arabic_text(page.page_content)
        full_text += text + "\n"
        page_map.append({
            "start": current_pos,
            "end": current_pos + len(text) + 1,
            "page": page.metadata.get("page", 0) + 1
        })
        current_pos += len(text) + 1

    article_pattern = re.compile(r"((?:مادة|المادة)\s*[^\d\s]*\s*([0-9٠-٩۰-۹]+)[^\d\s]*)")
    
    splits = list(article_pattern.finditer(full_text))
    documents_to_index = []
    
    if not splits:
        # Fallback to standard splitting if no articles found (safety)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents_to_index = text_splitter.split_documents(pages)
    else:
        for i in range(len(splits)):
            start_idx = splits[i].start()
            end_idx = splits[i+1].start() if i + 1 < len(splits) else len(full_text)
            
            chunk_text = full_text[start_idx:end_idx].strip()
            # Group 2 is the digits
            article_num_raw = splits[i].group(2)
            # Normalize digits to standard Western ones
            article_num = normalize_arabic_text(article_num_raw)
            # Extract digits
            article_num = "".join(re.findall(r"\d", article_num))
            
            
            if len(article_num) > 1:
                article_num = article_num[::-1]
            
            # Determine page number for this article's start
            page_num = "غير محدد"
            for p in page_map:
                if p["start"] <= start_idx < p["end"]:
                    page_num = p["page"]
                    break
            
            documents_to_index.append(Document(
                page_content=chunk_text,
                metadata={
                    "article": article_num if article_num else "غير محدد",
                    "page": page_num
                }
            ))
    
    # Save the vector store
    vector_store = FAISS.from_documents(documents_to_index, embeddings)
    vector_store.save_local(DB_FAISS_PATH)
    return vector_store

# Internal Query Rewriter Setup
def get_query_rewriter():
    template = """أنت محامي وخبير في قانون العمل المصري. حول سؤال المستخدم إلى كلمات مفتاحية قانونية دقيقة للبحث في نصوص القانون.
ركز على الموضوع الأساسي (مثل: ساعات العمل، الإجازات، عقد العمل، الأجور).
إذا ذكر المستخدم رقم مادة محدد (مثل: مادة 117)، يجب تضمين "مادة [الرقم]" بشكل صريح في الكلمات المفتاحية.
يجب كتابة أي أرقام في الاستعلام بالأرقام الإنجليزية (0-9).

سؤال المستخدم: {question}

كلمات البحث القانونية:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain

# FAQ Lookup Logic
def get_faq_context(question):
    try:
        faq_path = "faq.json"
        if not os.path.exists(faq_path):
            return ""
        
        with open(faq_path, "r", encoding="utf-8") as f:
            faq_data = json.load(f)
        
        normalized_q = normalize_arabic_text(question)
        matches = []
        
        for item in faq_data:
            # Check if any keyword or the question itself matches
            if any(keyword in normalized_q for keyword in item.get("keywords", [])):
                content = f"[قاعدة مؤكدة - مادة: {item['article']}, صفحة: {item['page']}]\n{item['answer']}"
                matches.append(content)
        
        if matches:
            return "--- معلومات مؤكدة من القواعد المتكررة ---\n" + "\n\n".join(matches) + "\n------------------------------------------\n"
        return ""
    except Exception as e:
        print(f"FAQ Error: {e}")
        return ""

# RAG Chain Setup
def get_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    rewriter = get_query_rewriter()
    
    template = """أنت مساعد قانوني ذكي ومتخصص جداً في قانون العمل المصري. 
استخدم المواد القانونية المسترجعة أدناه لتقديم إجابة مباشرة ودقيقة وصحيحة قانونياً على سؤال المستخدم.

تنبيهات هامة للبدء:
- "ساعات العمل الرسمية" أو "ساعات العمل الفعلية" (Article 117 عادة) تختلف عن "ساعات التواجد" أو "الفترة بين البداية والنهاية" (Article 119 عادة).
- إذا سأل المستخدم عن "ساعات العمل اليومية"، ابحث عن القاعدة التي تحدد الـ 8 ساعات.
- إذا سأل المستخدم عن "الحد الأقصى للتواجد" أو "متى ينتهي يوم العمل شاملاً الراحة"، ابحث عن قاعدة الـ 10 أو 12 ساعة.

القواعد الصارمة:
1. استخرج الحكم القانوني من المادة التي تتطابق تماماً مع استفسار المستخدم.
2. لا تخلط بين ساعات العمل الفعلية وساعات التواجد الإجمالية.
3. التزم تماماً بالأرقام الإنجليزية (0-9).
4. لا تضف أي استنتاجات شخصية.
5. إذا لم تجد الإجابة، استخدم الرد الافتراضي.

تنسيق الإجابة في حالة وجود معلومات:
الحکم القانوني:
<الإجابة مباشرة ودقيقة من النص مع استخدام الأرقام الإنجليزية>

رقم المادة:
[رقم المادة بالأرقام الإنجليزية]

رقم الصفحة:
[رقم الصفحة بالأرقام الإنجليزية]

تنسيق الرد الافتراضي:
الحکم القانوني:
لا توجد معلومات صريحة في قانون العمل المرفق للإجابة على هذا السؤال.


---
المواد القانونية المسترجعة:
{context}
---
سؤال المستخدم: {question}

إجابة المساعد القانوني بالعربية:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page", "غير محدد")
            article = doc.metadata.get("article", "غير محدد")
            content = f"[مادة: {article}, صفحة: {page}]\n{doc.page_content}"
            formatted.append(content)
        return "\n\n".join(formatted)

    # Modified chain to include rewriting step
    def run_chain(user_input):
        # Step 1: Normalize user input just in case
        normalized_user_input = normalize_arabic_text(user_input)
        # Step 2: Rewrite query for retrieval
        refined_query = rewriter.invoke({"question": normalized_user_input})
        # Step 3: Retrieve docs using refined query
        docs = retriever.invoke(refined_query)
        # Step 4: Format docs
        vector_context = format_docs(docs)
        
        # Step 5: Get FAQ context (Hybrid Layer)
        faq_context = get_faq_context(normalized_user_input)
        
        # Step 6: Combine contexts
        full_context = faq_context + "\n" + vector_context
        
        # Step 7: Final response using original question and combined context
        return prompt.invoke({"context": full_context, "question": user_input})

    chain = (
        RunnableLambda(run_chain)
        | llm
        | StrOutputParser()
    )
    return chain

# Load components
with st.spinner("جاري معالجة وثيقة القانون..."):
    vector_store = get_vector_store()
    rag_chain = get_rag_chain(vector_store)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("اسأل عن أي شيء في قانون العمل المصري..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في القانون..."):
            response = rag_chain.invoke(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})



