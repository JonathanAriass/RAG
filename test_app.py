"""
Simple test to verify the app is working
"""
import streamlit as st

st.title("🧾 Invoice Q&A Assistant - Test")
st.write("✅ Basic Streamlit is working!")

try:
    from config import AppConfig, get_app_title
    st.write("✅ Config module imported successfully")
    st.write(f"App title: {get_app_title()}")
except Exception as e:
    st.error(f"❌ Config import failed: {e}")

try:
    from database_manager import DatabaseManager
    st.write("✅ Database manager imported successfully")
except Exception as e:
    st.error(f"❌ Database manager import failed: {e}")

try:
    from document_processor import DocumentProcessor
    st.write("✅ Document processor imported successfully")
except Exception as e:
    st.error(f"❌ Document processor import failed: {e}")

try:
    from ui_components import CollectionSelector
    st.write("✅ UI components imported successfully")
except Exception as e:
    st.error(f"❌ UI components import failed: {e}")

try:
    from utils import create_collection_name
    st.write("✅ Utils imported successfully")
    st.write(f"Test collection name: {create_collection_name('test', 'invoice')}")
except Exception as e:
    st.error(f"❌ Utils import failed: {e}")

st.write("🔍 Testing main app import...")
try:
    # Try importing the main app class
    from app import InvoiceQAApp
    st.write("✅ Main app class imported successfully")
    st.write("🚀 All modules are working correctly!")
except Exception as e:
    st.error(f"❌ Main app import failed: {e}")
    st.write("There's an issue with the main app module.")