from typing import List

def merge(form: dict, table: List[dict]) -> dict:
    """Merge table into form"""
    final_result = form
    final_result["Table"] = table
    
    return final_result