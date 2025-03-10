#!/usr/bin/env python
"""
Generate Excel Template

This script generates an Excel template for the mechanical equipment classifier
training data with proper formatting, data validation, and instructions.
"""

import os
import pandas as pd
from pathlib import Path
import numpy as np

# Create directories if they don't exist
template_dir = Path("data/templates")
template_dir.mkdir(parents=True, exist_ok=True)

# Define the column headers based on the configuration
columns = [
    "equipment_tag",
    "manufacturer",
    "model",
    "category_name",
    "omniclass_code",
    "uniformat_code",
    "masterformat_code",
    "mcaa_system_category",
    "CategoryID",
    "OmniClassID",
    "UniFormatID",
    "MasterFormatID",
    "MCAAID",
    "LocationID",
    "service_life",
    "Precon_System",
    "Drawing_Abbreviation",
    "Precon_Tag",
    "System_Type_ID",
    "Equip_Name_ID",
    "Sub_System_ID",
    "Sub_System_Class",
    "Class_ID",
    "Unit"
]

# Define column descriptions for the instructions sheet
column_descriptions = {
    "equipment_tag": "Unique identifier for the equipment (e.g., EQ-001)",
    "manufacturer": "Equipment manufacturer (e.g., Trane, Carrier)",
    "model": "Equipment model number (e.g., CVHE-500)",
    "category_name": "Equipment category name (e.g., Chiller, Pump)",
    "omniclass_code": "OmniClass classification code (e.g., 23-33 13 13)",
    "uniformat_code": "Uniformat classification code (e.g., D3030.10)",
    "masterformat_code": "MasterFormat classification code (e.g., 23 64 16)",
    "mcaa_system_category": "MCAA system category (e.g., HVAC Equipment, Plumbing Equipment)",
    "CategoryID": "Category ID (integer)",
    "OmniClassID": "OmniClass ID (integer)",
    "UniFormatID": "Uniformat ID (integer)",
    "MasterFormatID": "MasterFormat ID (integer)",
    "MCAAID": "MCAA ID (e.g., H for HVAC, P for Plumbing)",
    "LocationID": "Location ID (integer)",
    "service_life": "Expected service life in years (float)",
    "Precon_System": "Precon system (e.g., Cooling, Heating)",
    "Drawing_Abbreviation": "Drawing abbreviation (e.g., CH for Chiller)",
    "Precon_Tag": "Precon tag (e.g., CH-01)",
    "System_Type_ID": "System type ID (e.g., H for HVAC)",
    "Equip_Name_ID": "Equipment name ID (e.g., CHLR for Chiller)",
    "Sub_System_ID": "Sub-system ID (e.g., CW for Chilled Water)",
    "Sub_System_Class": "Sub-system class (e.g., CENT for Centrifugal)",
    "Class_ID": "Class ID (integer)",
    "Unit": "Unit of measurement (e.g., tons, gpm)"
}

# Define sample data
sample_data = [
    ["EQ-001", "Trane", "CVHE-500", "Chiller", "23-33 13 13", "D3030.10", "23 64 16", "HVAC Equipment", 101, 2331313, "D303010", 236416, "H", 1, 20, "Cooling", "CH", "CH-01", "H", "CHLR", "CW", "CENT", 1, "tons"],
    ["EQ-002", "Carrier", "30XA-252", "Chiller", "23-33 13 13", "D3030.10", "23 64 16", "HVAC Equipment", 101, 2331313, "D303010", 236416, "H", 1, 15, "Cooling", "CH", "CH-02", "H", "CHLR", "CW", "AIR", 1, "tons"],
    ["EQ-003", "Bell & Gossett", "e-1510", "Pump", "23-21 13 13", "D3020.10", "23 21 23", "Hot Water Systems", 102, 2321313, "D302010", 232123, "H", 2, 10, "Heating", "P", "P-01", "H", "PUMP", "HW", "CENT", 2, "gpm"]
]

# Create a DataFrame with sample data
df = pd.DataFrame(sample_data, columns=columns)

# Define common values for data validation
mcaa_system_categories = [
    "HVAC Equipment",
    "Plumbing Equipment",
    "Mechanical/Sheetmetal",
    "Process Cooling Water",
    "Hot Water Systems",
    "Refrigeration",
    "Electrical",
    "Fire Protection",
    "Controls"
]

mcaa_ids = ["H", "P", "SM", "R", "E", "F", "C"]

# Create an Excel writer
excel_path = template_dir / "mechanical_equipment_training_data_template.xlsx"
writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

# Write the DataFrame to the Excel file
df.to_excel(writer, sheet_name='Training Data', index=False)

# Get the workbook and worksheet objects
workbook = writer.book
worksheet = writer.sheets['Training Data']

# Add a format for headers
header_format = workbook.add_format({
    'bold': True,
    'text_wrap': True,
    'valign': 'top',
    'fg_color': '#D7E4BC',
    'border': 1
})

# Add a format for the cells
cell_format = workbook.add_format({
    'border': 1
})

# Write the column headers with the header format
for col_num, value in enumerate(df.columns.values):
    worksheet.write(0, col_num, value, header_format)

# Set column widths
for i, column in enumerate(df.columns):
    column_width = max(len(column), df[column].astype(str).map(len).max())
    worksheet.set_column(i, i, column_width + 2)

# Add data validation for mcaa_system_category
mcaa_col = df.columns.get_loc("mcaa_system_category")
worksheet.data_validation(1, mcaa_col, 1000, mcaa_col, {
    'validate': 'list',
    'source': mcaa_system_categories
})

# Add data validation for MCAAID
mcaaid_col = df.columns.get_loc("MCAAID")
worksheet.data_validation(1, mcaaid_col, 1000, mcaaid_col, {
    'validate': 'list',
    'source': mcaa_ids
})

# Create an instructions sheet
instructions_sheet = workbook.add_worksheet('Instructions')

# Add a title
title_format = workbook.add_format({
    'bold': True,
    'font_size': 14,
    'align': 'center',
    'valign': 'vcenter',
    'fg_color': '#D7E4BC',
    'border': 1
})

instructions_sheet.merge_range('A1:B1', 'Mechanical Equipment Training Data Template', title_format)
instructions_sheet.set_column('A:A', 20)
instructions_sheet.set_column('B:B', 60)

# Add column descriptions
instructions_sheet.write(2, 0, 'Column Name', header_format)
instructions_sheet.write(2, 1, 'Description', header_format)

row = 3
for column, description in column_descriptions.items():
    instructions_sheet.write(row, 0, column, cell_format)
    instructions_sheet.write(row, 1, description, cell_format)
    row += 1

# Add general instructions
general_instructions = [
    "This template is for training data for the Mechanical Equipment Classifier.",
    "Fill in the data for each piece of equipment, one row per equipment.",
    "Required columns are marked with an asterisk (*) in the description.",
    "Use the Training Data sheet to enter your data.",
    "You can add as many rows as needed.",
    "Save the file as a CSV when you're done for use with the classifier."
]

instructions_sheet.merge_range(f'A{row+2}:B{row+2}', 'General Instructions', title_format)
row += 3
for instruction in general_instructions:
    instructions_sheet.merge_range(f'A{row}:B{row}', instruction, cell_format)
    row += 1

# Save the Excel file
writer.close()

print(f"Excel template created at {excel_path}")