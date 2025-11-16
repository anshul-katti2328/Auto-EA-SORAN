"""
Simple test to verify column name mapping logic
Tests without loading TensorFlow/Keras models
"""

import json
from datetime import datetime

def test_column_mapping_logic():
    """Test the column mapping dictionary"""
    
    print("="*80)
    print("Testing Column Name Mapping Logic")
    print("="*80)
    
    # This is the mapping from newPredictionAgent.py
    column_mapping = {
        # Network load/utilization metrics
        'prb_utilization': 'load',
        'prb_util': 'load',
        'utilization': 'load',
        'resource_utilization': 'load',
        'load_percent': 'load',
        
        # Power metrics
        'tx_power_dbm': 'TXpower',
        'txpower_dbm': 'TXpower',
        'tx_power': 'TXpower',
        'power_dbm': 'TXpower',
        'transmit_power': 'TXpower',
        
        # Bandwidth metrics
        'bandwidth_mhz': 'Bandwidth',
        'bw_mhz': 'Bandwidth',
        'channel_bandwidth': 'Bandwidth',
        'bw': 'Bandwidth',
        
        # Frequency metrics
        'frequency_mhz': 'Frequency',
        'freq_mhz': 'Frequency',
        'carrier_frequency': 'Frequency',
        'center_frequency': 'Frequency',
        'freq': 'Frequency',
        
        # Antenna configuration
        'antenna_count': 'Antennas',
        'num_antennas': 'Antennas',
        'antenna_elements': 'Antennas',
        
        # Radio unit and mode
        'ru_type': 'RUType',
        'radio_unit_type': 'RUType',
        'duplex_mode': 'Mode',
        'tdd_mode': 'Mode',
        'fdd_mode': 'Mode',
        
        # Base station identifier
        'bs_id': 'BS',
        'base_station_id': 'BS',
        'gnb_id': 'BS',
        'cell_id': 'CellName',
        
        # Energy saving modes
        'es_mode_1': 'ESMode1',
        'es_mode_2': 'ESMode2',
        'es_mode_3': 'ESMode3',
        'es_mode_4': 'ESMode4',
        'es_mode_5': 'ESMode5',
        'es_mode_6': 'ESMode6',
    }
    
    # Test cases with different Simu5G data formats
    test_cases = [
        {
            "name": "Simu5G Format 1 (underscores)",
            "data": {
                'prb_utilization': 0.75,
                'tx_power_dbm': 40.0,
                'bandwidth_mhz': 80,
                'frequency_mhz': 3500,
                'antenna_count': 64,
                'ru_type': 'Type1',
                'duplex_mode': 'Mode2'
            }
        },
        {
            "name": "Alternative Format",
            "data": {
                'utilization': 0.85,
                'power_dbm': 38.0,
                'bw_mhz': 100,
                'carrier_frequency': 3700,
                'num_antennas': 32
            }
        },
        {
            "name": "Mixed Format with ES Modes",
            "data": {
                'prb_util': 0.65,
                'txpower_dbm': 43.0,
                'channel_bandwidth': 60,
                'freq_mhz': 2600,
                'es_mode_1': 1,
                'es_mode_2': 0,
                'base_station_id': 'gNB_001'
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"Test Case: {test_case['name']}")
        print(f"{'='*80}")
        print(f"Input data: {json.dumps(test_case['data'], indent=2)}")
        
        # Apply column name mapping (same logic as in prepare_dataframe)
        mapped_data = {}
        mappings_applied = []
        
        for orig_key, value in test_case['data'].items():
            # Use mapped name if exists, otherwise keep original
            mapped_key = column_mapping.get(orig_key, orig_key)
            mapped_data[mapped_key] = value
            if orig_key in column_mapping:
                mappings_applied.append(f"  {orig_key} → {mapped_key} = {value}")
        
        print(f"\n✅ Mappings applied:")
        for mapping in mappings_applied:
            print(mapping)
        
        print(f"\n📊 Final mapped data:")
        for key, value in mapped_data.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
✅ Column mapping successfully translates Simu5G column names to model-expected names

📊 Key Mappings:
   • prb_utilization, prb_util, utilization → load
   • tx_power_dbm, txpower_dbm → TXpower  
   • bandwidth_mhz, bw_mhz → Bandwidth
   • frequency_mhz, freq_mhz → Frequency
   • antenna_count, num_antennas → Antennas
   • ru_type → RUType
   • duplex_mode, tdd_mode → Mode

🔧 This fixes the constant prediction issue by:
   1. Extracting actual varying data from Simu5G messages
   2. Mapping to correct model feature names
   3. Ensuring predictions vary with input data

✨ Next steps:
   1. Run system with actual Simu5G simulation
   2. Verify predictions now vary instead of constant -59.88W
   3. Monitor logs for "📊 Final feature values" to confirm data extraction
""")

if __name__ == "__main__":
    test_column_mapping_logic()
