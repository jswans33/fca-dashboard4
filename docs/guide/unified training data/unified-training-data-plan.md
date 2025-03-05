Here’s exactly what each of those reference files should contain to set you up
perfectly:

---

## 📂 **1\. MasterFormat (`masterformat.csv`)**

**Columns Needed:**

- `MasterFormat_Code`: (e.g., `23 31 13`)
- `MasterFormat_Title`: (e.g., `Metal Ducts`)
- `Description`: Plain-English description of equipment/system (e.g., "Metal
  HVAC ducts for air distribution")

**Example:**

| MasterFormat_Code | MasterFormat_Title | Description                       |
| ----------------- | ------------------ | --------------------------------- |
| 23 31 13          | Metal Ducts        | Metal HVAC ducts for air handling |

---

## 📂 **2\. UniFormat (`uniformat.csv`)**

**Columns Needed:**

- `UniFormat_Code`: (e.g., `D3050`)
- `UniFormat_Title`: (e.g., `HVAC Systems`)
- `Description`: Plain-English description of equipment/system (e.g., "Heating,
  Ventilation, and Air Conditioning")

---

## 📂 **3\. OmniClass (`omniclass.csv`)**

**Columns Needed:**

- `OmniClass_Code`: (e.g., `23-33 10 00`)
- `OmniClass_Title`: (e.g., `Air Distribution Systems`)
- `Description`: Plain-English description of equipment/system (e.g., "Ductwork
  and related equipment")

---

## 📂 **3\. MCAA (`mcaa.csv`)**

**Columns Needed:**

- `MCAA_Code`: (e.g., `15800`)
- `MCAA_Title`: (e.g., `Air Distribution Equipment`)
- `Description`: Plain-English description of equipment/system (e.g., "Metal
  duct systems, diffusers, dampers, etc.")

---

## 📌 **Recommendation for Each File:**

- Each file should be standardized CSVs with headers exactly as described above.
- Clearly separate each classification in its own CSV to ensure smooth joins
  later.
- Ensure consistency in the plain-English descriptions. These descriptions are
  crucial for matching and ML prediction later.

---

## **✅ Example CSV Row:**

**MasterFormat example (`masterformat.csv`):**

| MasterFormat_Code | MasterFormat_Title | Description               |
| ----------------- | ------------------ | ------------------------- |
| 23 31 13          | Metal Ducts        | Galvanized steel ductwork |

**UniFormat example (`uniformat.csv`):**

| UniFormat_Code | UniFormat_Title | Description                   |
| -------------- | --------------- | ----------------------------- |
| D3050          | HVAC Systems    | Ductwork and air distribution |

---

## 🔧 **How you'll use these files later:**

- You’ll **merge** these reference tables into a single unified reference data
  set based on the common field (`Description`).
- That unified reference will then be used to easily map or classify incoming
  equipment records from your engineers, giving your assets a complete
  classification set (`MasterFormat`, `UniFormat`, etc.).

---

## 🚩 **Quick Next Steps:**

- Populate the CSVs carefully.
- Keep descriptions straightforward and standardized.
- Save clearly named files, and you're ready to jump back into ETL & ML
  prediction!

---

Nope, you don’t **need** delimiters inside the descriptions, but keep these tips
in mind:

**✅ Best Practices:**

- **No Delimiters Needed:**  
  Just write simple, plain-English descriptions.  
  _(e.g., "Galvanized steel HVAC ductwork")_

- **Be Consistent:**  
  Keep wording standardized and straightforward. The clearer the match, the
  easier classification will be.

- **Avoid Commas if Possible:**  
  If your description naturally contains commas (e.g., lists), wrap the entire
  description in quotes so CSV parsing doesn't break.

  ```csv
  23 31 13,Metal Ducts,"Galvanized steel ducts, fittings, and accessories"
  ```

- **Avoid Special Characters:**  
  Avoid characters like `; | /` unless absolutely needed. If you use them,
  always quote the descriptions.

**🔹 Example of Good Descriptions (No delimiters required):**

```csv
MasterFormat_Code,MasterFormat_Title,Description
23 31 13,Metal Ducts,Galvanized steel ductwork
```

Or, if commas are unavoidable:

```csv
MasterFormat_Code,MasterFormat_Title,Description
23 31 13,Metal Ducts,"Galvanized steel ducts, fittings, and accessories"
```

---
