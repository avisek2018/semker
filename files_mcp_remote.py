from fastapi import FastAPI
import logging
import os
from pathlib import Path
import time
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="MCP File Server", description="MCP server for file operations over HTTP")

# Directory for storing text files
FILES_DIR = Path("./files")
FILES_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist

@app.get("/tools/list_files")
async def list_files() -> List[Dict[str, str]]:
    """
    List all text files in the files directory.
    Returns:
        List of dictionaries containing file name and path.
    """
    try:
        files = [f for f in FILES_DIR.iterdir() if f.is_file() and f.suffix == ".txt"]
        result = [{"name": f.name, "id": str(f)} for f in files]
        logger.info(f"Listed {len(result)} text files")
        return result
    except Exception as e:
        logger.error(f"Error listing text files: {str(e)}")
        raise

@app.post("/tools/create_file")
async def create_file(file_name: str, data: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Create a new text file with the provided name and data.
    Args:
        file_name (str): Name of the new text file.
        data (list): List of dictionaries containing arbitrary data (e.g., translation).
    Returns:
        Dictionary with status, message, file path, and file name.
    """
    try:
        # Ensure unique file name
        if not file_name.endswith('.txt'):
            timestamp = time.strftime("%Y%m%d%H%M%S")
            file_name = f"{file_name}_{timestamp}.txt"
        file_path = FILES_DIR / file_name
        
        # Extract headers dynamically from the first data item
        if not data or not isinstance(data, list) or not data[0]:
            logger.error("Invalid data format: Expected a non-empty list of dictionaries")
            return {
                "status": "error",
                "message": "Invalid data format: Expected a non-empty list of dictionaries",
                "file_id": "",
                "file_name": file_name
            }
        headers = list(data[0].keys())  # Use keys from the first dictionary
        
        # Format data as plain text
        content = ["Translation Data:", ",".join(headers)]
        for item in data:
            row = [str(item.get(key, '')) for key in headers]
            content.append(",".join(row))
        
        # Write to text file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        result = {
            "status": "success",
            "message": f"Created text file '{file_name}' at {file_path}",
            "file_id": str(file_path),
            "file_name": file_name
        }
        logger.info(f"Created text file: {file_name} (Path: {file_path})")
        return result
    except Exception as e:
        logger.error(f"Error creating text file: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to create text file: {str(e)}",
            "file_id": "",
            "file_name": file_name
        }

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 for accessibility
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")