import markdown
import pandas as pd


def parse_markdown_to_csv(input_markdown_file, output_csv_file):
    # Read markdown from file
    with open(input_markdown_file, "r", encoding="utf-8") as file:
        md_content = file.read()

    # Convert Markdown to HTML
    html = markdown.markdown(md_content, extensions=["tables"])

    # Use pandas to read HTML tables directly
    tables = pd.read_html(html)

    records = []

    for table in tables:
        # Skip the first row (header) and process the data
        for _, row in table.iloc[1:].iterrows():
            # Process the first set of columns (0, 1, 2)
            if pd.notna(row[0]) and row[0]:  # Check if key1 exists and is not NaN
                records.append(
                    {
                        "Keyword": str(row[0]).strip(),
                        "UF Number": str(row[1]).strip() if pd.notna(row[1]) else "",
                        "MF Number": str(row[2]).strip() if pd.notna(row[2]) else "",
                    }
                )

            # Process the second set of columns (3, 4, 5) if they exist
            if (
                len(row) > 3 and pd.notna(row[3]) and row[3]
            ):  # Check if key2 exists and is not NaN
                records.append(
                    {
                        "Keyword": str(row[3]).strip(),
                        "UF Number": str(row[4]).strip() if pd.notna(row[4]) else "",
                        "MF Number": str(row[5]).strip() if pd.notna(row[5]) else "",
                    }
                )

    # Create DataFrame from records and save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv_file, index=False)
    print(f"CSV created successfully at {output_csv_file}")


# Example usage:
input_markdown = r"C:\Repos\fca-dashboard4\nexusml\ingest\reference\uniformat\index\Index from UniFormat_redesign.pdf.md"
output_csv = "parsed_keywords.csv"

parse_markdown_to_csv(input_markdown, output_csv)
print("Markdown parsed and CSV created successfully.")
