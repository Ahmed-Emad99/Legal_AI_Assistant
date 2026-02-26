import os
import re
import unicodedata
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import streamlit as st

DB_FAISS_PATH = "vectorstore/db_faiss_v3"
PDF_PATH = "egyptian_labor_law.pdf"

def eastern_to_western(text):
    eastern_digits = '٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹'
    western_digits = '01234567890123456789'
    map_table = str.maketrans(eastern_digits, western_digits)
    return text.translate(map_table)

def normalize_arabic_text(text):
    if not text:
        return ""
    text = eastern_to_western(text)
    return unicodedata.normalize('NFKC', text)

@st.cache_resource
def get_vector_store_safe(embeddings):
    try:
        if os.path.exists(DB_FAISS_PATH):
            try:
                st.info("Loading existing FAISS vector store...")
                return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                st.warning(f"فشل تحميل الـ vector store الموجود، سيتم إعادة إنشائه: {e}")

        if not os.path.exists(PDF_PATH):
            st.error(f"ملف القانون غير موجود: {PDF_PATH}")
            st.stop()

        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()
        st.success(f"تم تحميل {len(pages)} صفحات من PDF بنجاح")

        # Prepare for splitting
        full_text = ""
        page_map = []
        current_pos = 0

        for i, page in enumerate(pages):
            try:
                text = normalize_arabic_text(page.page_content)
                full_text += text + "\n"
                page_map.append({"start": current_pos, "end": current_pos + len(text) + 1, "page": i+1})
                current_pos += len(text) + 1
            except Exception as e:
                st.warning(f"تعذر معالجة الصفحة {i+1}: {e}")

        # Article splitting
        article_pattern = re.compile(r"((?:مادة|المادة)\s*[^\d\s]*\s*([0-9٠-٩۰-۹]+)[^\d\s]*)")
        splits = list(article_pattern.finditer(full_text))
        documents_to_index = []

        if not splits:
            st.info("لم يتم العثور على مواد، سيتم استخدام تقسيم عام...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            documents_to_index = text_splitter.split_documents(pages)
        else:
            for i in range(len(splits)):
                try:
                    start_idx = splits[i].start()
                    end_idx = splits[i+1].start() if i + 1 < len(splits) else len(full_text)
                    chunk_text = full_text[start_idx:end_idx].strip()

                    article_num_raw = splits[i].group(2)
                    article_num = normalize_arabic_text(article_num_raw)
                    article_num = "".join(re.findall(r"\d", article_num))
                    if len(article_num) > 1:
                        article_num = article_num[::-1]

                    page_num = "غير محدد"
                    for p in page_map:
                        if p["start"] <= start_idx < p["end"]:
                            page_num = p["page"]
                            break

                    documents_to_index.append(Document(
                        page_content=chunk_text,
                        metadata={"article": article_num if article_num else "غير محدد",
                                  "page": page_num}
                    ))
                except Exception as e:
                    st.warning(f"خطأ أثناء معالجة المادة {i+1}: {e}")

        st.info(f"تم تجهيز {len(documents_to_index)} مستندات للفهرسة.")

        vector_store = FAISS.from_documents(documents_to_index, embeddings)
        vector_store.save_local(DB_FAISS_PATH)
        st.success("تم إنشاء وحفظ الـ vector store بنجاح")
        return vector_store

    except Exception as e:
        st.error(f"حدث خطأ أثناء إنشاء vector store: {e}")
        st.stop()
