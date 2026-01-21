#!/usr/bin/env python3
"""
Standalone script to generate Excel reports from existing analysis data.
Can be run independently or imported as a module.

Usage:
    python generate_excel_reports.py                    # Generate for all symbols
    python generate_excel_reports.py BTCUSDT            # Generate for specific symbol
    python generate_excel_reports.py BTCUSDT ETHUSDT    # Generate for multiple symbols
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.excel_report_generator import ExcelReportGenerator


def main():
    """Main function for standalone execution."""
    print("\n" + "="*80)
    print("CRYPTOCURRENCY EXCEL REPORT GENERATOR")
    print("="*80 + "\n")
    
    generator = ExcelReportGenerator()
    
    if len(sys.argv) > 1:
        # Generate for specific symbols
        symbols = sys.argv[1:]
        print(f"📊 Generating Excel reports for: {', '.join(symbols)}\n")
        
        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            print("-" * 40)
            
            reports = generator.collect_symbol_reports(symbol)
            
            if not reports:
                print(f"❌ No reports found for {symbol}")
                continue
            
            print(f"   Found {len(reports)} reports")
            
            df = generator.extract_data_for_excel(reports)
            print(f"   Extracted {len(df)} data points")
            
            excel_path = generator.create_excel_with_charts(symbol, df)
            
            if excel_path:
                print(f"   ✅ Excel report saved to: {excel_path}")
            else:
                print(f"   ❌ Failed to generate Excel report")
    else:
        # Generate for all symbols
        print("📊 Generating Excel reports for ALL symbols...\n")
        
        excel_files = generator.generate_all_reports()
        
        if excel_files:
            print(f"\n✅ Successfully generated {len(excel_files)} Excel reports:")
            print("-" * 80)
            for path in excel_files:
                print(f"   ✓ {path}")
        else:
            print("\n❌ No reports found to process")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
