import pytest
from bs4 import BeautifulSoup
from src.utils.processing_time import ProcessingTimer
from app.components.report_generator import ReportGenerator
from app.components.automatic_report_generation import AutomaticReportGenerator

# Mock objects required for AutomaticReportGenerator
class MockUser:
    def __init__(self):
        self.username = "test_user"

class MockSession:
    pass

class MockDB:
    def query(self, *args, **kwargs):
        return self
    def filter(self, *args, **kwargs):
        return self
    def all(self):
        return []

def test_processing_time_css_injection():
    """
    Test that ProcessingTimer injects the correct CSS variables for table backgrounds
    and text colors, overriding hardcoded colors.
    """
    css = ProcessingTimer._generate_css(is_dark_mode=False)
    assert "var(--background-color, #ffffff)" in css
    assert "var(--text-color, #111827)" in css
    assert "background-color: #333" not in css
    assert "background-color: #f5f5f5" not in css or "var(--secondary-background-color" in css
    
    css_dark = ProcessingTimer._generate_css(is_dark_mode=True)
    assert "var(--background-color, #ffffff)" in css_dark
    assert "var(--text-color, #111827)" in css_dark

def test_processing_time_render_expander_table_structure():
    """
    Test that the ProcessingTimer expander renders a valid HTML table and incorporates the new CSS.
    """
    # Create a mock Streamlit module
    class MockStreamlit:
        def __init__(self):
            self.markdown_calls = []
            
        class expander:
            def __init__(self, title, expanded):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
                
        def info(self, msg):
            pass
            
        def markdown(self, body, unsafe_allow_html=False):
            self.markdown_calls.append({"body": body, "unsafe": unsafe_allow_html})

    st_mock = MockStreamlit()
    timer = ProcessingTimer()
    
    with timer.time_block("Stage 1"):
        pass
    with timer.time_block("Stage 2"):
        pass
        
    ProcessingTimer.render_debug_expander(timer, is_dark_mode=False, st_module=st_mock)
    
    assert len(st_mock.markdown_calls) == 1
    call = st_mock.markdown_calls[0]
    assert call["unsafe"] is True
    
    html = call["body"]
    soup = BeautifulSoup(html, "html.parser")
    
    # Check that a table was rendered
    table = soup.find("table", class_="timing-table")
    assert table is not None
    
    # Check styles are embedded
    style = soup.find("style")
    assert style is not None
    assert "var(--background-color, #ffffff)" in style.string
    assert "var(--text-color, #111827)" in style.string

def test_report_generator_table_styles():
    """
    Test that the base ReportGenerator injects the correct Streamlit CSS variables
    to allow tables to adapt to Dark Mode, removing hardcoded 'background: white'.
    """
    generator = ReportGenerator(MockSession(), MockUser())
    report_html = generator.generate_report(target_user="test_target")
    
    soup = BeautifulSoup(report_html, "html.parser")
    style_tag = soup.find("style")
    
    assert style_tag is not None
    style_content = style_tag.string
    
    # Extract the table {} block
    import re
    table_block_match = re.search(r'table\s*\{([^\}]+)\}', style_content)
    assert table_block_match is not None, "table {} block missing from CSS"
    
    table_css = table_block_match.group(1)
    
    assert "background-color: var(--background-color, #ffffff)" in table_css
    assert "color: var(--text-color, #111827)" in table_css
    assert "background: white;" not in table_css

def test_report_generator_html_structure():
    """
    Verify that the tables in the ReportGenerator output are properly structured
    and actually affected by the style rules.
    """
    generator = ReportGenerator(MockSession(), MockUser())
    report_html = generator.generate_report(target_user="test_target")
    
    soup = BeautifulSoup(report_html, "html.parser")
    tables = soup.find_all("table")
    
    # Should have at least the System Information table
    assert len(tables) >= 1
    
    for table in tables:
        assert table.find("thead") is not None
        assert table.find("tbody") is not None

def test_automatic_report_generation_styles():
    """
    Test that the AutomaticReportGenerator uses dark-mode compatible CSS variables
    for all data tables, preventing unreadable white-on-white text in Dark Mode.
    """
    generator = AutomaticReportGenerator()
    
    # Test all default color schemes
    for template_id, template in generator.templates.items():
        css = generator._get_style_css(template.style_config)
        
        # Verify table styles use CSS vars
        assert "background-color: var(--background-color, #ffffff);" in css
        assert "color: var(--text-color, #111827);" in css
        
def test_automatic_report_generation_html_structure():
    """
    Test that automatic reports render the tables properly with the CSS variables.
    """
    generator = AutomaticReportGenerator()
    
    sample_data = {
        "documents": [{"name": "doc1.txt", "word_count": 100}],
        "violations": [{"policy": "no-copy", "severity": "High"}],
        "audit_trail": [{"user": "admin", "action": "login"}],
        "similarity_matrix": [{"doc1.txt": 1.0}]
    }
    
    for template_id in generator.templates:
        html = generator.generate_report(template_id, sample_data)
        soup = BeautifulSoup(html, "html.parser")
        
        # Ensure styles are included
        style = soup.find("style")
        assert style is not None
        assert "var(--background-color, #ffffff)" in style.string
        
        # Ensure tables are rendered
        tables = soup.find_all("table")
        assert len(tables) > 0
        for table in tables:
            # Automatic tables just use tr/th/td directly without thead/tbody in current implementation
            assert table.find("tr") is not None
