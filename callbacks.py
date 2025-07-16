"""
Callback handlers for the RAG Invoice Q&A Assistant
Contains custom callback handlers for progress tracking and monitoring
"""
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler

from config import AppConfig


class StreamlitProgressCallback(BaseCallbackHandler):
    """Custom Progress Callback Handler for LangChain with Streamlit"""
    
    def __init__(self, progress_bar, status_text, stage_weights: Optional[Dict[str, int]] = None):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.current_progress = 0
        self.stage_weights = stage_weights or AppConfig.PROGRESS_WEIGHTS
        self._token_count = 0
    
    def on_retriever_start(self, serialized: Dict[str, Any], query: str, **kwargs: Any) -> None:
        """Called when retriever starts"""
        try:
            self.status_text.text("🔍 Searching through invoice documents...")
            self.current_progress += self.stage_weights['retrieval']
            self.progress_bar.progress(min(self.current_progress, 95))
        except Exception:
            pass
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts"""
        try:
            self.status_text.text("🤖 Starting AI analysis...")
            self.current_progress += self.stage_weights['llm_start']
            self.progress_bar.progress(min(self.current_progress, 95))
        except Exception:
            pass
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when LLM generates a new token"""
        try:
            self._token_count += 1
            
            # Update progress based on token generation
            if self._token_count % 5 == 0:  # Update every 5 tokens to avoid too frequent updates
                token_progress = min(self._token_count / 200, 1.0) * self.stage_weights['llm_processing']
                self.progress_bar.progress(min(self.current_progress + token_progress, 95))
        except Exception:
            # Silently handle Streamlit API exceptions
            pass
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when LLM ends"""
        try:
            self.status_text.text("✅ Analysis complete!")
            self.current_progress += self.stage_weights['completion']
            self.progress_bar.progress(100)
        except Exception:
            pass
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM encounters an error"""
        try:
            self.status_text.text(f"❌ LLM Error: {str(error)}")
            self.progress_bar.progress(0)
        except Exception:
            pass
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain starts"""
        try:
            self.status_text.text("🔗 Starting processing chain...")
        except Exception:
            pass
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain ends"""
        try:
            self.status_text.text("✅ Processing chain complete!")
        except Exception:
            pass
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when chain encounters an error"""
        try:
            self.status_text.text(f"❌ Chain Error: {str(error)}")
            self.progress_bar.progress(0)
        except Exception:
            pass


class LoggingCallback(BaseCallbackHandler):
    """Callback for logging chain execution details"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logs = []
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log chain start"""
        if self.verbose:
            print(f"[CHAIN_START] {serialized.get('name', 'Unknown')}")
            print(f"[INPUTS] {inputs}")
        
        self.logs.append({
            "event": "chain_start",
            "name": serialized.get('name', 'Unknown'),
            "inputs": inputs
        })
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log chain end"""
        if self.verbose:
            print(f"[CHAIN_END] Outputs: {outputs}")
        
        self.logs.append({
            "event": "chain_end",
            "outputs": outputs
        })
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Log LLM start"""
        if self.verbose:
            print(f"[LLM_START] Model: {serialized.get('name', 'Unknown')}")
            print(f"[PROMPTS] {prompts}")
        
        self.logs.append({
            "event": "llm_start",
            "model": serialized.get('name', 'Unknown'),
            "prompts": prompts
        })
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Log LLM end"""
        if self.verbose:
            print(f"[LLM_END] Response: {response}")
        
        self.logs.append({
            "event": "llm_end",
            "response": response
        })
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logged events"""
        return self.logs
    
    def clear_logs(self) -> None:
        """Clear all logs"""
        self.logs = []