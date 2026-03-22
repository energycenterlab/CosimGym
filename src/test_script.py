"""
Test Script for COSIM Gym Framework

This script demonstrates how to run HELICS federation simulations
with different progress monitoring configurations.

QUICK START:
  python test_script.py                    # Run with progress bar (default)
  
MODIFY OPTIONS:
  1. Enable/disable progress bar by uncommenting desired option in main block
  2. Use advanced examples for custom configurations
  3. See configuration reference at bottom for all options

Author: Pietro Rando Mazzarino
Date: 2025
"""

from core.ScenarioManager import main, ScenarioManager


# main('simple_test', enable_progress_bar=True)  # Run with progress bar (default)
# main('simple_test_multifederations', enable_progress_bar=False) # Run without progress bar (maximum performance)



# OSMSES26 - working examples
main('bui_hp_test_base', enable_progress_bar=False) # Run without progress bar (maximum performance)
main('pv_batt_test_base', enable_progress_bar=False) # Run without progress bar (maximum performance)



















'''
# =====================================================================
# BASIC USAGE - Choose one option below
# =====================================================================

if __name__ == "__main__":
    """
    COSIM Gym Test Script - Scenario Simulation
    
    This script runs a simple test scenario demonstrating the framework functionality.
    Uncomment the option you want to use.
    """
    
    # ─────────────────────────────────────────────────────────────────
    # OPTION 1: Standard simulation with progress bar (RECOMMENDED)
    # ─────────────────────────────────────────────────────────────────
    # Shows real-time federation progress with simulation time tracking
    # Use for: development, debugging, demonstrations
    # Performance: ~3-8% overhead
    
    # main('simple_test', enable_progress_bar=False)
    # main('simple_test_multifederations', enable_progress_bar=False)
    main('bui_hp_test_base', enable_progress_bar=False)
    
    
    # ─────────────────────────────────────────────────────────────────
    # OPTION 2: High-performance simulation (no progress bar)
    # ─────────────────────────────────────────────────────────────────
    # Maximum performance with minimal monitoring overhead
    # Use for: production runs, performance testing, batch processing
    # Performance: 0% overhead (fastest execution)
    
    # main('simple_test', enable_progress_bar=False)


# =====================================================================
# ADVANCED USAGE EXAMPLES (uncomment to use)
# =====================================================================

def example_custom_configuration():
    """
    Example: Custom progress bar configuration
    
    This shows how to configure the HELICS query system for different
    monitoring requirements and performance needs.
    """
    
    with ScenarioManager('simple_test') as manager:
        # Configure progress monitoring parameters
        manager.configure_query_frequency(
            enabled=True,           # Enable progress monitoring
            frequency_ms=1000,      # Query every 1 second (1 Hz)
            adaptive=False,         # Use fixed frequency (not adaptive)
            timeout_ms=500          # 500ms timeout per query
        )
        
        print("Running with custom 1-second query interval...")
        manager.simulate()


def example_adaptive_frequency():
    """
    Example: Adaptive frequency based on simulation time steps
    
    The system automatically calculates optimal query frequency
    based on your federation's time step configuration.
    """
    
    with ScenarioManager('simple_test') as manager:
        # Enable adaptive frequency (default behavior)
        manager.configure_query_frequency(
            enabled=True,
            adaptive=True,          # Auto-adjust based on time steps
            timeout_ms=500
        )
        
        # Show calculated frequency
        adaptive_freq = manager.get_adaptive_query_frequency()
        print(f"Using adaptive frequency: {adaptive_freq:.3f}s ({1/adaptive_freq:.1f} queries/sec)")
        
        manager.simulate()


def example_performance_comparison():
    """
    Example: Compare performance with different monitoring settings
    
    This demonstrates the performance impact of different configurations.
    """
    import time
    
    print("🔍 Performance Comparison Test")
    print("-" * 40)
    
    # Test 1: With progress bar
    print("Test 1: WITH progress monitoring...")
    start_time = time.time()
    main('simple_test', enable_progress_bar=True)
    with_progress_time = time.time() - start_time
    
    # Test 2: Without progress bar
    print("Test 2: WITHOUT progress monitoring...")
    start_time = time.time()
    main('simple_test', enable_progress_bar=False)
    without_progress_time = time.time() - start_time
    
    # Analysis
    overhead = with_progress_time - without_progress_time
    overhead_percent = (overhead / without_progress_time) * 100 if without_progress_time > 0 else 0
    
    print("\nResults:")
    print(f"  With progress:    {with_progress_time:.3f}s")
    print(f"  Without progress: {without_progress_time:.3f}s")
    print(f"  Overhead:         {overhead:.3f}s ({overhead_percent:.1f}%)")


# =====================================================================
# CONFIGURATION REFERENCE
# =====================================================================

"""
📝 PROGRESS BAR CONFIGURATION OPTIONS:

Basic Control:
  main('scenario_name', enable_progress_bar=True/False)  # Simple on/off
  manager.enable_progress_bar()     # Enable with defaults
  manager.disable_progress_bar()    # Disable completely

Advanced Configuration:
  manager.configure_query_frequency(
      enabled=True,                 # Enable/disable monitoring
      frequency_ms=500,             # Query interval (100-2000ms)
      adaptive=True,                # Auto-adjust based on time steps
      timeout_ms=500                # Query timeout (100-5000ms)
  )

Performance Guidelines:
  • Disabled:           0% overhead      (fastest)
  • Low freq (2000ms):  ~1-3% overhead  (minimal monitoring)
  • Default (500ms):    ~3-8% overhead  (balanced)
  • High freq (100ms):  ~8-15% overhead (detailed monitoring)

Use Cases:
  • Development/Debug:   enable_progress_bar=True (default settings)
  • Production Runs:     enable_progress_bar=False (maximum performance)
  • Long Simulations:    Custom config with 1-2 second intervals
  • Demonstrations:      enable_progress_bar=True with adaptive=True

Example Scenarios:
  # Quick development test
  main('simple_test')
  
  # Production batch run
  main('simple_test', enable_progress_bar=False)
  
  # Custom monitoring for long simulation
  with ScenarioManager('scenario') as manager:
      manager.configure_query_frequency(frequency_ms=2000, adaptive=False)
      manager.simulate()
"""

# =====================================================================
# UNCOMMENT TO RUN EXAMPLES
# =====================================================================

# example_custom_configuration()
# example_adaptive_frequency()  
# example_performance_comparison()
'''