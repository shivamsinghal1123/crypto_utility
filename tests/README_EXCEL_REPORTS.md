# Excel Report Generator

Automated Excel report generation system for cryptocurrency analysis data. This utility collects all analysis reports and generates comprehensive Excel files with charts and visualizations.

## Features

### 📊 Data Included
- **Market Data**: Current price, 24h high/low, volume, price changes
- **Technical Analysis**: RSI, MACD, trend strength, momentum scores
- **Fundamental Analysis**: Overall scores, tokenomics, team, technology, adoption
- **Sentiment Analysis**: News sentiment, social sentiment, Fear & Greed index
- **Price Predictions**: Forecasted prices, support/resistance levels, probabilities
- **24-Hour Trading Range**: Complete intraday price analysis

### 📈 Excel Sheets Generated

1. **Dashboard** - Summary overview with latest metrics and statistics
2. **Data** - Complete raw data table
3. **Price Analysis** - Price charts and volume analysis
4. **Technical Analysis** - RSI, MACD, trend indicators with charts
5. **Fundamental Analysis** - Scoring metrics with comparison charts
6. **Sentiment Analysis** - Sentiment trends and Fear & Greed index
7. **Predictions** - Price forecasts and probability charts

### 🎨 Visualization Features
- Line charts for price movements and trends
- Bar charts for volume analysis
- Multi-series charts for comparative analysis
- Formatted headers and data cells
- Auto-adjusted column widths

## Installation

Install required dependencies:

```bash
pip install pandas openpyxl
```

## Usage

### Automatic Integration

The Excel report generator is **automatically integrated** with the main analysis workflow. Every time you run an analysis, the Excel file is automatically updated:

```bash
python main.py --symbol SOLUSDT
```

After analysis completes, check the `crypto_analyzer/tests/` directory for the Excel file:
- `SOLUSDT_analysis.xlsx`

### Manual Generation

#### Generate for All Symbols

```bash
cd crypto_analyzer/tests
python generate_excel_reports.py
```

This will:
1. Scan all reports in `data/reports/`
2. Group by symbol
3. Generate separate Excel files for each symbol

#### Generate for Specific Symbol

```bash
python generate_excel_reports.py BTCUSDT
```

#### Generate for Multiple Symbols

```bash
python generate_excel_reports.py BTCUSDT ETHUSDT SOLUSDT
```

### Programmatic Usage

```python
from tests.excel_report_generator import ExcelReportGenerator

# Initialize generator
generator = ExcelReportGenerator()

# Generate for all symbols
excel_files = generator.generate_all_reports()

# Generate for specific symbol
reports = generator.collect_symbol_reports('BTCUSDT')
df = generator.extract_data_for_excel(reports)
excel_path = generator.create_excel_with_charts('BTCUSDT', df)

# Update with new report
from pathlib import Path
import json

report_path = Path('data/reports/BTCUSDT_20260121_120000/report.json')
with open(report_path) as f:
    report_data = json.load(f)

excel_path = generator.update_excel_with_new_report('BTCUSDT', report_data)
```

## File Structure

```
crypto_analyzer/
├── tests/
│   ├── excel_report_generator.py      # Main Excel generator class
│   ├── generate_excel_reports.py      # Standalone utility script
│   ├── BTCUSDT_analysis.xlsx          # Generated Excel files
│   ├── ETHUSDT_analysis.xlsx
│   └── SOLUSDT_analysis.xlsx
└── data/
    └── reports/
        ├── BTCUSDT_20260121_120000/
        │   └── report.json
        ├── SOLUSDT_20260121_115348/
        │   └── report.json
        └── ...
```

## Output File Location

Excel files are saved in: `crypto_analyzer/tests/`

File naming convention: `{SYMBOL}_analysis.xlsx`

Examples:
- `BTCUSDT_analysis.xlsx`
- `ETHUSDT_analysis.xlsx`
- `SOLUSDT_analysis.xlsx`

## Chart Types

### Price Analysis Sheet
- **Price Movement Chart**: Line chart showing current price, 24h high, and 24h low
- **Volume Chart**: Bar chart displaying 24-hour trading volume

### Technical Analysis Sheet
- **RSI Indicator**: Line chart tracking RSI values over time
- **Trend & Momentum**: Combined chart for trend strength and momentum scores

### Fundamental Analysis Sheet
- **Fundamental Scores**: Multi-series line chart showing all scoring metrics

### Sentiment Analysis Sheet
- **Sentiment Analysis**: Overall, news, and social sentiment trends
- **Fear & Greed Index**: Separate chart for market sentiment index

### Predictions Sheet
- **Current vs Forecasted Price**: Comparison chart
- **Support & Resistance Levels**: Key price levels visualization

## Data Updates

### Automatic Updates
When you run `python main.py --symbol SYMBOL`:
1. Analysis is performed
2. Report is saved to `data/reports/`
3. Excel file is automatically updated with new data
4. All charts are regenerated with complete dataset

### Manual Updates
Run the standalone script to regenerate all Excel files:
```bash
python tests/generate_excel_reports.py
```

## Troubleshooting

### Missing Dependencies
```bash
pip install pandas openpyxl
```

### No Reports Found
- Ensure reports exist in `data/reports/` directory
- Check that report folders follow naming convention: `{SYMBOL}_{TIMESTAMP}`
- Verify `report.json` files exist in report folders

### Excel File Not Created
- Check file permissions in `tests/` directory
- Ensure pandas and openpyxl are installed correctly
- Check logs for error messages

### Charts Not Displaying
- Open Excel file
- Charts should be embedded in respective sheets
- If missing, check that data columns are not empty

## Advanced Features

### Custom Output Directory

```python
generator = ExcelReportGenerator(
    reports_dir='/custom/path/to/reports',
    output_dir='/custom/path/to/output'
)
```

### Data Filtering

```python
# Collect specific date range
reports = generator.collect_symbol_reports('BTCUSDT')
df = generator.extract_data_for_excel(reports)

# Filter by date
df_filtered = df[df['datetime'] >= '2026-01-15']

# Regenerate Excel
excel_path = generator.create_excel_with_charts('BTCUSDT', df_filtered)
```

## Notes

- Excel files use `.xlsx` format (not `.xls`)
- All timestamps are preserved from original reports
- Data is sorted chronologically
- Empty/missing values are handled gracefully
- Charts auto-scale based on data range

## Example Output

After running analysis for SOLUSDT multiple times, the Excel file will contain:
- **Dashboard**: Latest price, changes, forecasts, summary statistics
- **Data**: Full table with 30+ columns of metrics
- **6 Chart Sheets**: Each with relevant visualizations

## Integration with Main Workflow

The Excel generator is called automatically in `main.py` after each analysis:

```python
# In main.py, after saving JSON/TXT reports:
excel_path = self.excel_generator.update_excel_with_new_report(symbol, report)
```

This ensures your Excel files are always up-to-date with the latest analysis data.
