Here’s a structured JSON schema template that you can use to upload your
UniFormat PDF to Llama Parse to extract relevant data for classification
purposes:

### JSON Schema for UniFormat Data Extraction

```json
{
  "UniFormat": {
    "type": "object",
    "properties": {
      "Sections": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "Number": { "type": "string" },
            "Title": { "type": "string" },
            "MasterFormat_Number": { "type": "string" },
            "Explanation": { "type": "string" },
            "Includes": { "type": "array", "items": { "type": "string" } },
            "Performance_Requirements": {
              "type": "array",
              "items": { "type": "string" }
            },
            "Subsections": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "Number": { "type": "string" },
                  "Title": { "type": "string" },
                  "MasterFormat_Number": { "type": "string" },
                  "Includes": {
                    "type": "array",
                    "items": { "type": "string" }
                  },
                  "Performance_Requirements": {
                    "type": "array",
                    "items": { "type": "string" }
                  }
                },
                "required": ["Number", "Title"]
              }
            }
          },
          "required": ["Number", "Title"]
        }
      }
    },
    "required": ["Sections"]
  }
}
```

### Explanation of Schema:

- **Sections**: Represents each major section from the UniFormat document.
  - **Number**: UniFormat code number (e.g., D1010.10).
  - **Title**: Description or title of the section (e.g., Elevators).
  - **MasterFormat_Number**: Corresponding MasterFormat number.
  - **Explanation**: Additional details from the section.
  - **Includes**: List of items explicitly mentioned within the section.
  - **Performance_Requirements**: References to performance requirements if
    mentioned.
  - **Subsections**: Further granularity of the classification, structured
    similarly to the main sections.

### How to Use this Schema:

1. **Llama Parse Setup**:

   - Upload this schema to Llama Parse.
   - Set it to extract content from the provided PDF, adhering to this schema.

2. **Data Extraction**:

   - Run extraction on your PDFs (like the one you uploaded).
   - The output JSON will give you structured data ready for database ingestion
     or further processing.

3. **Integration**:
   - Load the structured JSON into your data model (e.g., using Python with
     Pandas).
   - Use it to classify equipment or as a reference lookup.

This structure ensures you accurately capture all critical classification
elements to utilize the model effectively.

---

The prompts you've set are quite clear, but here's how you could slightly
enhance them for precision:

### Improved Model Prompts:

**System Prompt:**

```
Output the provided content formatted as a complete LaTeX document, including necessary preamble and structure.
```

**Append to System Prompt:**

```
Do not format headings as titles. Instead, prefix headings explicitly with "TITLE:" followed by the heading text.
```

**User Prompt:**

```
Also, translate the entire content into French.
```

This setup ensures clarity on exactly what the model should produce and how it
should handle titles explicitly.
