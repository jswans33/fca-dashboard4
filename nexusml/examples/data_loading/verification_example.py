"""
Example script for using the verification utility with the Medtronics pipeline.

This script demonstrates how to:
1. Run the Medtronics pipeline
2. Verify the pipeline output
3. Generate a verification report
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fca_dashboard.pipelines import MedtronicsPipeline
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.verification_util import (
    verify_database_table,
    generate_verification_report,
    verify_pipeline_output,
)


def main():
    """Run the example."""
    logger = get_logger("verification_example")
    logger.info("Starting verification example")
    
    # Create and run the Medtronics pipeline
    pipeline = MedtronicsPipeline()
    results = pipeline.run(clear_output=True, preserve_db=True)
    
    if results["status"] != "success":
        logger.error(f"Pipeline failed: {results['message']}")
        return 1
    
    # Get the verification results from the pipeline output
    if "verification" in results["data"]:
        verification_results = results["data"]["verification"]
        logger.info("Verification results from pipeline run:")
        
        # Check for placeholder values in usassetid column
        if "verifications" in verification_results:
            if "no_placeholder_values" in verification_results["verifications"]:
                placeholder_results = verification_results["verifications"]["no_placeholder_values"]
                
                if "usassetid" in placeholder_results:
                    usassetid_results = placeholder_results["usassetid"]
                    logger.info(f"usassetid column:")
                    logger.info(f"  Total rows: {usassetid_results['total_rows']}")
                    logger.info(f"  Placeholder count: {usassetid_results['placeholder_count']}")
                    logger.info(f"  Placeholder percentage: {usassetid_results['placeholder_percentage'] * 100:.2f}%")
                    
                    if usassetid_results['value_counts']:
                        logger.info(f"  Placeholder value counts:")
                        for value, count in usassetid_results['value_counts'].items():
                            logger.info(f"    '{value}': {count}")
    
    # You can also run verification separately
    logger.info("\nRunning verification separately:")
    
    # Get the database path from the pipeline results
    db_path = results["data"]["db_path"]
    
    # Define verification configuration
    verification_config = {
        "no_placeholder_values": {
            "columns": ["usassetid", "asset tag"],
            "placeholder_values": ["no id", "none", "n/a", "unknown", ""]
        },
        "no_null_values": {
            "columns": ["usassetid", "asset tag", "asset name"]
        }
    }
    
    # Get analysis and validation results from the pipeline
    analysis_results = pipeline.analysis_results
    validation_results = pipeline.validation_results
    
    # Verify the database table with analysis and validation results
    verification_results = verify_pipeline_output(
        pipeline_name="medtronics",
        db_name=os.path.basename(db_path),
        table_name="asset_data",
        verification_config=verification_config,
        output_dir=os.path.dirname(db_path),
        analysis_results=analysis_results,
        validation_results=validation_results
    )
    
    # Generate a verification report
    report = generate_verification_report(verification_results)
    logger.info(f"\nVerification Report:\n{report}")
    
    # Save the report to a file
    output_dir = os.path.dirname(db_path)
    report_path = os.path.join(output_dir, "verification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Verification report saved to {report_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())