import pytest
import os
from pathlib import Path


from main import (
    parse_xml,
    create_table,
)


@pytest.fixture(scope="session", autouse=True)
def setup_env():
    # Set up environment variables
    os.environ["XML_ROOT"] = str(Path(__file__).resolve().parent.joinpath("tests"))
    os.environ[
        "URL_SHAS"
    ] = '{"tt-forge-fe": "0872c4955e50be6b38a15635e1ea2e00189d45fe", "tt-torch": "15c37e0fdf54e791555c0a1bda2cc8ba922d7a0f"}'


def test_parse_xml():
    model_tests, card_archs = parse_xml()
    
    # Test basic structure
    assert isinstance(model_tests, dict), f"model_tests should be a dict, got {type(model_tests)}"
    assert isinstance(card_archs, set), f"card_archs should be a set, got {type(card_archs)}"
    assert len(model_tests) > 0, "model_tests should not be empty"
    
    print(f"\nFound {len(model_tests)} models")
    print(f"Found card architectures: {card_archs}\n")
    
    # Test card architectures
    assert len(card_archs) > 0, "card_archs should not be empty"

    all_valid_cards = {"N/A", "Wormhole", "Blackhole"}
    assert card_archs.issubset(all_valid_cards) or card_archs == all_valid_cards   
    
    # Test model entries
    for model_name, tests in model_tests.items():
        assert isinstance(tests, list), f"Tests for {model_name} should be a list"
        
        for test in tests:
            assert isinstance(test, dict), f"Test entry for {model_name} should be a dict"
            
            # Check required fields
            required_fields = {'model_name', 'card', 'frontend', 'status', 'file_path'}
            missing_fields = required_fields - set(test.keys())
            assert not missing_fields, f"Missing required fields in {model_name}: {missing_fields}"
            
            # Validate field values
            assert test['status'] == 'PASSED', f"Test status for {model_name} should be PASSED, got {test['status']}"
            assert test['frontend'] in ['tt-forge-fe', 'tt-torch', 'tt-xla'], \
                f"Invalid frontend for {model_name}: {test['frontend']}"
            
            # Validate card values
            assert test['card'] in all_valid_cards, \
                f"Invalid card for {model_name}: {test['card']}"
            
            # Validate file path
            assert test['file_path'].endswith(".py"), \
                f"Invalid file path for {model_name}: {test['file_path']}"
            

def test_create_table():
    model_tests, card_archs = parse_xml()
    
    # Call create_table
    table_data = create_table(model_tests, card_archs)
    
    # Verify table structure
    assert isinstance(table_data, list), "Table data should be a list"
    assert len(table_data) > 0, "Table data should not be empty"
    
    # Verify header
    header = table_data[0]
    assert header[0] == "frontend", "First column should be frontend"
    assert header[1] == "model_name", "Second column should be model_name"
    assert set(header[2:]) == card_archs, "Remaining columns should be card architectures"
    
    # Verify data rows
    data_rows = table_data[1:]
    assert len(data_rows) >= 3, "Should have at least 3 rows of data"
    
    # Check that each row has correct structure and valid values
    for row in data_rows:
        # Check frontend is valid
        assert row[0] in {"tt-forge-fe", "tt-torch", "tt-xla"}, f"Invalid frontend: {row[0]}"
        
        # Check model name is formatted as markdown link
        assert row[1].startswith("[") and "]" in row[1], f"Model name not formatted as markdown link: {row[1]}"
        
        # Check status indicators
        wormhole_idx = header.index("Wormhole") if "Wormhole" in header else None
        blackhole_idx = header.index("Blackhole") if "Blackhole" in header else None
        
        if wormhole_idx is not None:
            assert row[wormhole_idx] in {"✅", "N/A"}, f"Invalid Wormhole status: {row[wormhole_idx]}"
        
        if blackhole_idx is not None:
            assert row[blackhole_idx] in {"✅", "N/A"}, f"Invalid Blackhole status: {row[blackhole_idx]}"
    



if __name__ == "__main__":
    pytest.main(['-v', __file__])  # Run tests normally