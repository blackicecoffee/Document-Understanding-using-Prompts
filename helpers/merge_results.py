from typing import List, Optional

def merge(information: dict, table: Optional[List[dict]]) -> dict:
    """Merge table into form"""
    final_result = information
    if table:
        final_result["Table"] = table
    
    return final_result