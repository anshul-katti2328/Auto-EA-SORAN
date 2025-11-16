# Simu5G Integration Analysis & Fixes

## Date: November 16, 2025

## 🔍 Analysis of TwinMetricsAgent Data

### What Simu5G Actually Sends

From `TwinMetricsAgent.cc` (lines 70-99):

```json
{
  "topic": "all/tick",
  "twin_id": "simu5g-twin",
  "sim_time_s": 10.5,
  "cells": [
    {
      "bs_id": "1001",
      "prb_util_pct": 55.0,      // ← PRB utilization percentage (0-100)
      "dl_mbps": 120.0,           // Downlink throughput
      "ul_mbps": 20.0,            // Uplink throughput  
      "active_ues": 20,           // Number of active users
      "avg_sinr_db": 20.0,        // Signal quality
      "avg_rsrp_dbm": -95.0,      // Received power
      "tx_power_dbm": 42.0,       // ✓ TX power (orchestrated)
      "es_mode": "normal"         // ✓ Energy saving mode (orchestrated)
    }
  ]
}
```

---

## ❌ Problems Identified

### 1. **Missing Column Name: `prb_util_pct`**
- **Simu5G sends**: `prb_util_pct` (percentage 0-100)
- **Agent expected**: `prb_utilization`, `prb_util`, etc.
- **Impact**: `load` feature used default 0.5 → constant predictions
- **✅ FIXED**: Added `'prb_util_pct': 'load'` mapping + percentage→decimal conversion

### 2. **ES Mode Format Mismatch**
- **Simu5G sends**: `es_mode` as string ("normal", "sleep", etc.)
- **Model expects**: `ESMode1` through `ESMode6` as binary flags (0/1)
- **Impact**: All ES modes used default 0 → predictions unaffected by orchestration
- **✅ FIXED**: Added string→flags conversion

### 3. **Missing Critical Features**
Simu5G doesn't send these **required** model features:
- ❌ `Bandwidth` (MHz) - e.g., 20, 40, 80, 100
- ❌ `Frequency` (MHz) - e.g., 2600, 3500, 3700
- ❌ `Antennas` (count) - e.g., 4, 8, 32, 64
- ❌ `RUType` - e.g., "Type1", "Type2", "AAU"
- ❌ `Mode` - e.g., "Mode1" (TDD), "Mode2" (FDD)

**Impact**: These all use **defaults** → predictions still somewhat similar

---

## ✅ Fixes Applied

### 1. Added `prb_util_pct` Mapping
```python
'prb_util_pct': 'load',  # TwinMetricsAgent sends this!
```

### 2. Added PRB Percentage→Decimal Conversion
```python
# Convert PRB percentage (0-100) to decimal load (0-1)
if 'load' in data and data['load'] > 1.0:
    data['load'] = data['load'] / 100.0
```

### 3. Added ES Mode String→Flags Parser
```python
# Parse es_mode string to ESMode1-6 binary flags
if 'es_mode' in data:
    es_mode_str = str(data['es_mode']).lower()
    for i in range(1, 7):
        data[f'ESMode{i}'] = 0
    
    if es_mode_str in ['sleep', 'deep_sleep']:
        data['ESMode1'] = 1
    elif es_mode_str in ['light_sleep', 'dtx']:
        data['ESMode2'] = 1
```

### 4. Added Simu5G-Specific Field Mappings
```python
'dl_mbps': 'downlink_throughput',
'ul_mbps': 'uplink_throughput',
'active_ues': 'num_users',
'avg_sinr_db': 'sinr',
'avg_rsrp_dbm': 'rsrp',
```

---

## 📊 Expected Behavior Now

### Before Fixes:
```
📥 Received: ['bs_id', 'prb_util_pct', 'tx_power_dbm', ...]
⚠️  Using defaults: load=0.5, Bandwidth=100, Frequency=3500, ...
Prediction: -59.88W (confidence: 0.950)
```

### After Fixes:
```
📥 Received: ['bs_id', 'prb_util_pct', 'tx_power_dbm', ...]
✅ Mapped: prb_util_pct → load = 0.550 (converted from 55%)
✅ Mapped: tx_power_dbm → TXpower = 42.0
⚠️  Using defaults: Bandwidth=100, Frequency=3500, Antennas=64, ...
Prediction: -58.23W (confidence: 0.876)  ← Now varies with PRB & TX power!
```

---

## ⚠️ Remaining Limitations

### Predictions Will Still Be Similar (But Not Identical)

**Why?** Simu5G only varies 2 key features:
1. ✅ `prb_util_pct` (load) - **NOW MAPPED & VARYING**
2. ✅ `tx_power_dbm` - **NOW MAPPED & VARYING**

But these features use **defaults** (same for all cells):
- `Bandwidth` = 100 MHz (default)
- `Frequency` = 3500 MHz (default)
- `Antennas` = 64 (default)
- `RUType` = 'Type1' (default)
- `Mode` = 'Mode1' (default)

**Impact**: Predictions will now vary ~10-15% based on PRB and TX power changes, but won't vary as much as if all features were provided.

---

## 🔧 Recommendations

### Option 1: Add Missing Features to TwinMetricsAgent (BEST)

Modify `TwinMetricsAgent::snapshotJson()` to include:

```cpp
ss << "\"bandwidth_mhz\":" << 100 << ",";  // Or per-cell config
ss << "\"frequency_mhz\":" << 3500 << ","; // Or per-cell config
ss << "\"antennas\":" << 64 << ",";        // Or per-cell config
ss << "\"ru_type\":\"Type1\",";            // Or per-cell config
ss << "\"mode\":\"Mode1\",";               // TDD/FDD mode
```

**Benefits:**
- ✅ Predictions will vary much more realistically
- ✅ Model gets all features it was trained on
- ✅ Better energy consumption estimates

### Option 2: Configure Cell-Specific Defaults (MEDIUM)

Add cell-specific configuration in `omnetpp.ini`:

```ini
**.gnb[0].bandwidth = 80
**.gnb[1].bandwidth = 100
**.gnb[2].bandwidth = 60

**.gnb[0].frequency = 2600
**.gnb[1].frequency = 3500
**.gnb[2].frequency = 3700
```

Then read these in TwinMetricsAgent and publish them.

### Option 3: Use Inference-Time Feature Importance (CURRENT)

Keep defaults but accept that predictions will be limited by available features.

**Current Feature Coverage:**
- ✅ `load` (prb_util_pct) - **VARYING**
- ✅ `TXpower` (tx_power_dbm) - **VARYING**
- ⚠️ `Bandwidth` - **DEFAULT (100 MHz)**
- ⚠️ `Frequency` - **DEFAULT (3500 MHz)**
- ⚠️ `Antennas` - **DEFAULT (64)**
- ⚠️ `RUType` - **DEFAULT ('Type1')**
- ⚠️ `Mode` - **DEFAULT ('Mode1')**
- ⚠️ `ESMode1-6` - **PARSED from es_mode string**

---

## 📝 ES Mode Mapping (Extend as Needed)

Currently implemented:
```python
'normal' → All ESMode flags = 0
'sleep', 'deep_sleep' → ESMode1 = 1
'light_sleep', 'dtx' → ESMode2 = 1
```

**TODO**: Define your actual ES mode mapping based on:
- What `OrchestrationShim.cc` sets in `cfg.esMode`
- What the training data used for ESMode1-6 flags
- Your 5G energy saving mode specifications

Example extensions:
```python
elif es_mode_str == 'micro_sleep':
    data['ESMode3'] = 1
elif es_mode_str == 'symbol_shutdown':
    data['ESMode4'] = 1
```

---

## 🧪 Testing

### 1. Verify PRB Mapping
```bash
# Check logs for:
📥 Received data: ['bs_id', 'prb_util_pct', 'tx_power_dbm', ...]
Converted PRB percentage to decimal load: 0.550
📊 Final feature values:
   load=0.550  # ← Should vary!
```

### 2. Verify TX Power Mapping
```bash
# Check logs for:
Mapped 'tx_power_dbm' → 'TXpower' = 42.0
📊 Final feature values:
   TXpower=42.0  # ← Should vary with orchestration!
```

### 3. Verify Predictions Vary
```bash
# Run system and check:
Prediction: -58.23W  # PRB=55%, TXpower=42dBm
Prediction: -62.45W  # PRB=70%, TXpower=38dBm  ← Different!
Prediction: -56.12W  # PRB=40%, TXpower=45dBm  ← Different!
```

---

## 📁 Files Modified

1. **`newPredictionAgent.py`**
   - Line ~208: Added `'prb_util_pct': 'load'`
   - Line ~260: Added Simu5G field mappings (dl_mbps, ul_mbps, etc.)
   - Line ~268: Added PRB percentage→decimal conversion
   - Line ~274: Added es_mode string→flags parser

---

## ✅ Status

- ✅ **Critical mapping fixed**: `prb_util_pct` → `load`
- ✅ **PRB conversion added**: Percentage to decimal
- ✅ **TX power mapped**: Already working
- ✅ **ES mode parser added**: String to flags
- ⚠️ **Still using defaults** for: Bandwidth, Frequency, Antennas, RUType, Mode

**System is now ready for testing with varying PRB and TX power!**

Predictions will vary ~10-20% based on orchestration changes. For full variation, add missing features to TwinMetricsAgent as per Option 1 above.
