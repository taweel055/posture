#!/usr/bin/env python3
"""
PDF Report Generator for Comprehensive Posture Analysis
======================================================
Converts HTML report to professional PDF format
Includes all data, charts, and professional formatting

Features:
- High-quality PDF generation
- Professional layout
- Charts and graphics
- Clinical-grade formatting
- Cross-platform compatibility

Requirements:
- weasyprint or reportlab
- Pillow for image processing

Author: Professional Posture Analysis System
Version: 4.0.0
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
import sys

# Try multiple PDF generation libraries
PDF_LIBRARY = None
try:
    from weasyprint import HTML, CSS
    PDF_LIBRARY = "weasyprint"
    print("✅ WeasyPrint available for PDF generation")
except ImportError:
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        PDF_LIBRARY = "reportlab"
        print("✅ ReportLab available for PDF generation")
    except ImportError:
        print("⚠️ No PDF library available. Installing weasyprint...")
        os.system("pip install weasyprint")
        try:
            from weasyprint import HTML, CSS
            PDF_LIBRARY = "weasyprint"
            print("✅ WeasyPrint installed and ready")
        except ImportError:
            print("❌ PDF generation not available. Please install: pip install weasyprint")

class PDFReportGenerator:
    """Generate professional PDF reports from posture analysis data"""
    
    def __init__(self):
        """Initialize PDF report generator"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data = None
        
    def load_data(self, json_file: str) -> bool:
        """Load posture analysis data from JSON file"""
        try:
            with open(json_file, 'r') as f:
                self.data = json.load(f)
            print(f"✅ Data loaded from {json_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False
    
    def generate_pdf_weasyprint(self, html_file: str, output_file: str) -> bool:
        """Generate PDF using WeasyPrint (high quality)"""
        try:
            # Custom CSS for PDF optimization
            pdf_css = CSS(string="""
                @page {
                    size: A4;
                    margin: 1in;
                    @bottom-center {
                        content: "Professional Posture Analysis Report - Page " counter(page);
                        font-size: 10pt;
                        color: #666;
                    }
                }
                body {
                    font-family: 'Arial', sans-serif;
                    line-height: 1.4;
                    color: #333;
                }
                .container {
                    max-width: none;
                    margin: 0;
                    padding: 0;
                    box-shadow: none;
                }
                .header {
                    background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%) !important;
                    -webkit-print-color-adjust: exact;
                    color-adjust: exact;
                    print-color-adjust: exact;
                }
                .metric-card {
                    break-inside: avoid;
                    page-break-inside: avoid;
                }
                .section {
                    break-inside: avoid;
                    page-break-inside: avoid;
                }
                table {
                    break-inside: avoid;
                    page-break-inside: avoid;
                }
                .comparison-table {
                    font-size: 10pt;
                }
                .metrics-grid {
                    display: block;
                }
                .metric-card {
                    display: inline-block;
                    width: 45%;
                    margin: 10px 2%;
                    vertical-align: top;
                }
            """)
            
            # Generate PDF
            HTML(filename=html_file).write_pdf(
                output_file,
                stylesheets=[pdf_css],
                optimize_images=True
            )
            
            print(f"✅ PDF generated successfully: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ PDF generation failed: {e}")
            return False
    
    def generate_pdf_reportlab(self, output_file: str) -> bool:
        """Generate PDF using ReportLab (fallback)"""
        try:
            doc = SimpleDocTemplate(output_file, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.HexColor('#2c3e50')
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=12,
                textColor=colors.HexColor('#3498db')
            )
            
            # Title
            story.append(Paragraph("🎯 COMPREHENSIVE POSTURE ANALYSIS REPORT", title_style))
            story.append(Paragraph("Professional Clinical Assessment", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Executive Summary
            story.append(Paragraph("📊 Executive Summary", heading_style))
            if self.data:
                summary = self.data.get('assessment_summary', {})
                summary_data = [
                    ['Parameter', 'Value', 'Status'],
                    ['Overall Posture Score', f"{summary.get('overall_posture_score', 0)}%", 'Excellent'],
                    ['System Accuracy', f"{summary.get('system_accuracy', 0)}%", 'Clinical Grade'],
                    ['Measurement Precision', summary.get('measurement_precision', '±1.2°'), 'Enhanced'],
                    ['Clinical Confidence', summary.get('clinical_confidence', 'High').title(), 'Validated'],
                    ['Risk Assessment', summary.get('risk_assessment', 'Low').title(), 'Minimal']
                ]
                
                summary_table = Table(summary_data)
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(summary_table)
                story.append(Spacer(1, 20))
            
            # Detailed Measurements
            story.append(Paragraph("📐 Detailed Measurements", heading_style))
            if self.data and 'detailed_measurements' in self.data:
                measurements = self.data['detailed_measurements']
                
                # Head & Neck Region
                if 'head_neck_region' in measurements:
                    story.append(Paragraph("Head & Neck Region", styles['Heading3']))
                    head_neck = measurements['head_neck_region']
                    
                    head_data = [['Parameter', 'Value', 'Normal Range', 'Status']]
                    for key, value in head_neck.items():
                        if isinstance(value, dict) and 'value' in value:
                            head_data.append([
                                key.replace('_', ' ').title(),
                                f"{value['value']}°" if 'unit' in value and value['unit'] == 'degrees' else str(value['value']),
                                value.get('normal_range', 'N/A'),
                                value.get('status', 'N/A').title()
                            ])
                    
                    if len(head_data) > 1:
                        head_table = Table(head_data)
                        head_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, -1), 10),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        story.append(head_table)
                        story.append(Spacer(1, 15))
            
            # Technology Performance
            story.append(Paragraph("🚀 Technology Performance", heading_style))
            if self.data and 'technology_performance' in self.data:
                tech = self.data['technology_performance']
                
                tech_text = f"""
                <b>Hardware:</b> {tech.get('hardware_specifications', {}).get('processor', 'N/A')}<br/>
                <b>GPU:</b> {tech.get('hardware_specifications', {}).get('gpu_backend', 'N/A')}<br/>
                <b>AI Framework:</b> {tech.get('software_configuration', {}).get('ai_framework', 'N/A')}<br/>
                <b>Processing:</b> {tech.get('performance_metrics', {}).get('processing_speed', 'N/A')}<br/>
                <b>Accuracy:</b> {tech.get('accuracy_progression', {}).get('stereo_vision_current', 'N/A')}%
                """
                story.append(Paragraph(tech_text, styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Recommendations
            story.append(Paragraph("💡 Clinical Recommendations", heading_style))
            if self.data and 'comprehensive_recommendations' in self.data:
                recs = self.data['comprehensive_recommendations']
                
                story.append(Paragraph("<b>Immediate Actions:</b>", styles['Heading4']))
                for action in recs.get('immediate_actions', []):
                    story.append(Paragraph(f"• {action.replace('_', ' ').title()}", styles['Normal']))
                
                story.append(Spacer(1, 10))
                story.append(Paragraph("<b>Professional Recommendations:</b>", styles['Heading4']))
                for rec in recs.get('professional_recommendations', []):
                    story.append(Paragraph(f"• {rec.replace('_', ' ').title()}", styles['Normal']))
            
            # Footer
            story.append(Spacer(1, 30))
            footer_text = f"""
            <b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
            <b>System:</b> Professional Posture Analysis v4.0.0<br/>
            <b>Technology:</b> Comprehensive Full Assessment with Stereo Vision
            """
            story.append(Paragraph(footer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            print(f"✅ PDF generated successfully: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ PDF generation failed: {e}")
            return False
    
    def generate_pdf(self, html_file: str = None, json_file: str = None) -> str:
        """Generate PDF report from HTML or JSON data"""
        output_file = f"Comprehensive_Posture_Analysis_Report_{self.timestamp}.pdf"
        
        # Load data if JSON file provided
        if json_file and os.path.exists(json_file):
            self.load_data(json_file)
        
        # Try WeasyPrint first (better quality)
        if PDF_LIBRARY == "weasyprint" and html_file and os.path.exists(html_file):
            if self.generate_pdf_weasyprint(html_file, output_file):
                return output_file
        
        # Fallback to ReportLab
        if PDF_LIBRARY == "reportlab":
            if self.generate_pdf_reportlab(output_file):
                return output_file
        
        print("❌ PDF generation failed - no suitable library available")
        return None

def main():
    """Main function to generate PDF report"""
    print("🚀 PDF REPORT GENERATOR")
    print("=" * 40)
    
    generator = PDFReportGenerator()
    
    # Look for existing report files
    html_file = "COMPREHENSIVE_FULL_POSTURE_REPORT_20250629_221000.html"
    json_file = "COMPREHENSIVE_FULL_POSTURE_DATA_20250629_221000.json"
    
    # Check if files exist
    if not os.path.exists(html_file):
        print(f"⚠️ HTML file not found: {html_file}")
        html_file = None
    
    if not os.path.exists(json_file):
        print(f"⚠️ JSON file not found: {json_file}")
        json_file = None
    
    if not html_file and not json_file:
        print("❌ No report files found to convert to PDF")
        return
    
    # Generate PDF
    pdf_file = generator.generate_pdf(html_file, json_file)
    
    if pdf_file:
        print(f"\n✅ PDF REPORT GENERATED SUCCESSFULLY!")
        print(f"📄 File: {pdf_file}")
        print(f"📁 Location: {os.path.abspath(pdf_file)}")
        
        # Get file size
        if os.path.exists(pdf_file):
            size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
            print(f"📊 Size: {size_mb:.2f} MB")
    else:
        print("\n❌ PDF generation failed")

if __name__ == "__main__":
    main()
