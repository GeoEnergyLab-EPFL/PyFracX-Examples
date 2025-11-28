#!/bin/bash

# Configuration
TIMEOUT_SECONDS=3600  # Maximum time per script in seconds
LOG_DIR="logs"       # Directory to store log files

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Array of Python scripts to run
scripts=(
    "2D-uncoupled/Constant-Friction-Constant-Permeability/ConstantOverpressure.py"
    "2D-uncoupled/Weakening-Friction-Constant-Permeability/ConstantOverpressure-LinearWeakening.py"
    "2D-coupled/Constant-Friction-Constant-Permeability/ConstantOverpressure.py"
    "2D-coupled/HydraulicFracture-reopening/ConstantRate-HF-reopening.py"
    "Axisymmetry-uncoupled/Constant-Friction-Constant-Permeability/ConstantRate-ConstantFriction-CriticallyStressed.py"
    "Axisymmetry-uncoupled/Constant-Friction-Constant-Permeability/ConstantRate-SlipWeakeningFriction.py"
    "Axisymmetry-uncoupled/Constant-Friction-Constant-Permeability/ConstantRate-ConstantFriction-MarginallyPressurised.py"
    "Axisymmetry-coupled/Constant-Friction-Constant-Permeability/ConstantRate-ConstantFriction-CriticallyStressed.py"
    "Axisymmetry-coupled/Constant-Friction-Constant-Permeability/ConstantRate-SlipWeakeningFriction.py"
    "Axisymmetry-coupled/Constant-Friction-Constant-Permeability/ConstantRate-ConstantFriction-shutin.py"
    "Axisymmetry-coupled/Constant-Friction-Constant-Permeability/ConstantRate-ConstantFriction-MarginallyPressurised.py"
    "Axisymmetry-coupled/HydraulicFracture-reopening/run_simulation.py"
    "Axisymmetry-coupled/Weakening-Friction-Variable-Permeability/ConstantRate-Dilatancy-BBSpring.py"
    "Axisymmetry-coupled/Basel/AF2_1_smooth.py"
    "3D-uncoupled/Constant-Friction-Constant-Permeability/ConstantRate-ConstantFriction.py"
    "3D-uncoupled/Weakening-Friction-Constant-Permeability/ConstantRate-SlipWeakeningFriction.py"
    "3D-coupled/Constant-Friction-Constant-Permeability/ConstantRate_ConstantFriction_coupled.py"
    "3D-coupled/Constant-Friction-Constant-Permeability/ConstantRate_ConstantFriction_coupled_poly_mesh.py"
    "3D-coupled/hydraulic_fracture_reopening/hydraulic_fracture_reopening_M_poly_mesh.py"
)

# Loop over each script
echo "Starting to run ${#scripts[@]} Python scripts..."
echo "Timeout per script: ${TIMEOUT_SECONDS}s"
echo "Log directory: ${LOG_DIR}"
echo "----------------------------------------"

# Function to run script with timeout and signal handling
run_with_timeout() {
    local script="$1"
    local log_file="$2"
    local timeout_sec="$3"

    # Run Python process in background and capture its PID
    python "$script" > "$log_file" 2>&1 &
    local pid=$!

    # Setup trap to forward SIGINT to child process
    trap "kill -INT $pid 2>/dev/null" INT

    # Wait for process with timeout
    local count=0
    while kill -0 $pid 2>/dev/null; do
        if [ $count -ge $timeout_sec ]; then
            kill -TERM $pid 2>/dev/null
            sleep 1
            kill -KILL $pid 2>/dev/null
            trap - INT
            return 124  # Timeout exit code
        fi
        sleep 1
        ((count++))
    done

    # Get exit status
    wait $pid
    local exit_code=$?
    trap - INT
    return $exit_code
}

for script in "${scripts[@]}"; do
    # Generate log filename from script path (replace / with _)
    log_file="${LOG_DIR}/$(echo "$script" | sed 's/\//_/g' | sed 's/\.py$/.log/')"

    echo "Running: $script"
    echo "Log file: $log_file"

    # Run the script with timeout and signal handling
    run_with_timeout "$script" "$log_file" "$TIMEOUT_SECONDS"

    # Check exit status
    exit_status=$?
    if [ $exit_status -eq 124 ]; then
        echo "  ⚠ TIMEOUT after ${TIMEOUT_SECONDS}s" | tee -a "$log_file"
    elif [ $exit_status -eq 130 ]; then
        echo "  ⚠ Interrupted by user (SIGINT)"
        break
    elif [ $exit_status -eq 0 ]; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed with exit code $exit_status"
    fi
    echo ""
done

echo "----------------------------------------"
echo "All scripts completed. Logs are in: ${LOG_DIR}/"
