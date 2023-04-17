{
    "class": "ScrapedData",
    "description": "A section of text",
    "properties": [
        {"name": "title", "dataType": ["string"], "description": "Title of the section"},
        {"name": "url", "dataType": ["string"], "description": "The url something was scraped from"},
        {"name": "text", "dataType": ["string"], "description": "Text of the section"},
        {"name": "summary", "dataType": ["string"], "description": "Summary of the section"},
        {"name": "tokens", "dataType": ["int"], "description": "Token count of the section"},
        {"name": "sentiment", "dataType": ["number"], "description": "The sentiment rating of the section"},
        {"name": "cost", "dataType": ["float"], "description": "Cost of processing the section"},
        {
            "name": "hasCost",
            "dataType": ["Cost"],
            "description": "Relation between the section and its cost",
            "cardinality": "ToOne",
        },
    ],
},
{
    "class": "Cost",
    "description": "Cost of processing a section",
    "properties": [
        {"name": "amount", "dataType": ["float"], "description": "Amount of cost"},
    ],
}