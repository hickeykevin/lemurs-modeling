import streamlit as st
import os
from pathlib import Path
import subprocess
import signal
import time
import yaml
from typing import Any
st.set_page_config(page_title="Lemurs Modeling GUI", layout="wide")

st.title("Lemurs Modeling - Hydra GUI")

CONFIG_DIR = Path("configs")
LIST_GROUPS = {"preprocessors", "callbacks", "logger"}

def load_defaults_from_config(config_path: Path) -> dict:
    """Reads the defaults block from a Hydra config yaml and returns a dict mapping group to default value."""
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        defaults = data.get("defaults", [])
        group_defaults = {}
        for item in defaults:
            if isinstance(item, dict):
                for k, v in item.items():
                    group_defaults[k] = v
        return group_defaults
    except Exception as e:
        return {}

def load_raw_yaml(group: str, name: str) -> dict:
    """Helper to load a YAML file without resolving defaults."""
    path = CONFIG_DIR / group / f"{name}.yaml"
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def load_config_with_defaults(group: str, name: str) -> dict:
    """Recursively loads config YAML files to resolve defaults/inheritance, returning raw parameters."""
    if not name or name == "None" or name == "null" or name is None:
        return {}
    
    path = CONFIG_DIR / group / f"{name}.yaml"
    if not path.exists():
        return {}
        
    try:
        with open(path, "r") as f:
            content = yaml.safe_load(f) or {}
            
        resolved = {}
        
        # Resolve any defaults referenced in this file
        defaults = content.get("defaults", [])
        if isinstance(defaults, list):
            for item in defaults:
                if isinstance(item, str) and item != "_self_":
                    # Inherit from a file in the same directory, e.g. 'default'
                    resolved.update(load_config_with_defaults(group, item))
                elif isinstance(item, dict):
                    # Nested or external defaults, e.g., net: lstm
                    for k, v in item.items():
                        # If this nested group corresponds to a subdirectory (a subgroup),
                        # skip resolving it here to avoid duplicating it in parent controls.
                        if not (CONFIG_DIR / group / k).is_dir():
                            nested_params = load_config_with_defaults(f"{group}/{k}", v)
                            for nk, nv in nested_params.items():
                                resolved[f"{k}.{nk}"] = nv
                        
        # Overwrite with current file's parameters, ignoring hydra-specific metadata
        for k, v in content.items():
            if k not in ("defaults", "_target_") and not k.startswith("_"):
                resolved[k] = v
                
        return resolved
    except Exception as e:
        return {}


def render_parameter_editor(param_path: str, original_val: Any, widget_key_prefix: str, config_overrides: dict):
    """Recursively renders form fields for configuration parameters (handling nested dicts/lists)."""
    if isinstance(original_val, dict):
        label = param_path.split('.')[-1]
        with st.container(border=True):
            st.markdown(f"**{label}**")
            for k, v in original_val.items():
                render_parameter_editor(f"{param_path}.{k}", v, f"{widget_key_prefix}_{k}", config_overrides)
    else:
        label = param_path.split('.')[-1]
        if isinstance(original_val, bool):
            val = st.checkbox(label, value=original_val, key=widget_key_prefix)
        elif isinstance(original_val, int):
            val = st.number_input(label, value=original_val, step=1, key=widget_key_prefix)
        elif isinstance(original_val, float):
            val = st.number_input(label, value=original_val, step=0.01, format="%.5f", key=widget_key_prefix)
        elif isinstance(original_val, list):
            val_str = st.text_input(label, value=str(original_val), key=widget_key_prefix)
            try:
                import ast
                val = ast.literal_eval(val_str)
            except Exception:
                val = val_str
        else:
            val = st.text_input(label, value=str(original_val), key=widget_key_prefix)
            
        # Compare as string to detect changes cleanly
        if str(val) != str(original_val):
            config_overrides[param_path] = val


def render_config_group_ui(group: str, selected_val: str, config_overrides: dict, selected_configs: dict):
    """Recursively renders form fields for configuration parameters and subgroup selection."""
    if selected_val in ("None", "null", None):
        return
        
    # 1. Render leaf parameters of the current group
    params = load_config_with_defaults(group, selected_val)
    if params:
        for param_name, original_val in params.items():
            widget_key = f"param_{group.replace('/', '_')}_{selected_val}_{param_name}"
            render_parameter_editor(f"{group.replace('/', '.')}.{param_name}", original_val, widget_key, config_overrides)
            
    # 2. Render subgroups (subdirectories)
    group_dir = CONFIG_DIR / group
    if group_dir.exists():
        for subdir in sorted([d for d in group_dir.iterdir() if d.is_dir()]):
            sub_group = f"{group}/{subdir.name}"
            sub_options = ["None"] + get_options_for_group(subdir)
            
            # Find default for subgroup from parent config's defaults block
            sub_default_val = None
            main_config = load_raw_yaml(group, selected_val)
            main_defaults = main_config.get("defaults", [])
            if isinstance(main_defaults, list):
                for item in main_defaults:
                    if isinstance(item, dict) and subdir.name in item:
                        sub_default_val = item[subdir.name]
                        
            is_list_default = isinstance(sub_default_val, list) or subdir.name in LIST_GROUPS
            
            # Render subgroup selection and settings inside a collapsed expander
            with st.expander(f"📁 {sub_group} settings", expanded=False):
                st.markdown(f"**Select {subdir.name}**")
                if is_list_default:
                    if isinstance(sub_default_val, list):
                        default_options = [x for x in sub_default_val if x in sub_options]
                    elif sub_default_val in sub_options:
                        default_options = [sub_default_val]
                    else:
                        default_options = []
                        
                    selected_sub = st.multiselect(
                        f"Select {subdir.name}", 
                        sub_options, 
                        default=default_options, 
                        label_visibility="collapsed", 
                        key=f"select_{group.replace('/', '_')}_{selected_val}_{subdir.name}"
                    )
                    
                    if selected_sub:
                        selected_configs[sub_group] = selected_sub
                        # Render parameter editors for each item in the multiselect
                        for val in selected_sub:
                            st.markdown(f"⚙️ **{val} parameters**")
                            st.markdown('<div style="padding-left: 20px; border-left: 2px dashed #666; margin-top: 10px; margin-bottom: 20px;">', unsafe_allow_html=True)
                            render_config_group_ui(sub_group, val, config_overrides, selected_configs)
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    sub_default_idx = 0
                    if sub_default_val in sub_options:
                        sub_default_idx = sub_options.index(sub_default_val)
                    elif str(sub_default_val).lower() == "null" and "null" in sub_options:
                        sub_default_idx = sub_options.index("null")
                        
                    selected_sub = st.selectbox(
                        f"Select {subdir.name}", 
                        sub_options, 
                        index=sub_default_idx, 
                        label_visibility="collapsed", 
                        key=f"select_{group.replace('/', '_')}_{selected_val}_{subdir.name}"
                    )
                    
                    if selected_sub != "None":
                        selected_configs[sub_group] = selected_sub
                        # Indent subgroup parameters and nested elements using a visual tree line
                        st.markdown('<div style="padding-left: 20px; border-left: 2px dashed #666; margin-top: 10px; margin-bottom: 20px;">', unsafe_allow_html=True)
                        render_config_group_ui(sub_group, selected_sub, config_overrides, selected_configs)
                        st.markdown('</div>', unsafe_allow_html=True)


def get_options_for_group(group_path: Path):
    """Returns a list of options (yaml filenames without extension) in a folder"""
    if not group_path.exists() or not group_path.is_dir():
        return ["default"]
    
    options = []
    for f in group_path.glob("*.yaml"):
        options.append(f.stem)
    
    # Try to make 'default' the first option if it exists
    if "default" in options:
        options.remove("default")
        return ["default"] + sorted(options)
    elif "null" in options:
        options.remove("null")
        return ["null"] + sorted(options)
    else:
        return sorted(options)

# Main Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Configuration")
    
    mode = st.radio("Execution Mode", ["Train", "Cross Validate", "Evaluate"])
    if mode == "Train":
        script = "src/train.py"
    elif mode == "Cross Validate":
        script = "src/cv_train.py"
    else:
        script = "src/eval.py"
    
    st.subheader("Execution Environment")
    env = st.radio("Environment", ["Local", "Slurm Cluster"])
    
    srun_prefix = []
    if env == "Slurm Cluster":
        with st.expander("🖥️ Slurm Resources Config", expanded=True):
            partition = st.text_input("Partition", value="short")
            cpus = st.number_input("CPU Cores (-c)", value=2, min_value=1)
            gpus = st.number_input("GPUs", value=0, min_value=0)
            mem = st.number_input("Memory (MB) (--mem)", value=8192, min_value=256, step=256)
            time_limit = st.number_input("Time limit (minutes) (-t)", value=120, min_value=1)
            
            srun_prefix = ["srun", "-p", partition, "-c", str(cpus), f"--mem={mem}", "-t", str(time_limit)]
            if gpus > 0:
                srun_prefix.append(f"--gpus={gpus}")
                
    selected_configs = {}
    config_overrides = {}
    
    # Load defaults from configs/train.yaml, configs/cv_train.yaml or configs/eval.yaml
    if mode == "Train":
        config_file = "configs/train.yaml"
    elif mode == "Cross Validate":
        config_file = "configs/cv_train.yaml"
    else:
        config_file = "configs/eval.yaml"
    group_defaults = load_defaults_from_config(Path(config_file))
    
    # Custom group ordering: data, model, trainer, callbacks, then the rest
    custom_order = ["data", "model", "trainer", "callbacks"]
    ordered_keys = sorted(
        group_defaults.keys(),
        key=lambda x: custom_order.index(x) if x in custom_order else len(custom_order) + list(group_defaults.keys()).index(x)
    )
    group_defaults = {k: group_defaults[k] for k in ordered_keys}
    
    st.subheader("Core Groups")
    for group, default_val in group_defaults.items():
        group_dir = CONFIG_DIR / group
        if group_dir.exists():
            options = get_options_for_group(group_dir)
            if "None" not in options and "null" not in options:
                options = ["None"] + options
                
            is_list_default = isinstance(default_val, list) or group in LIST_GROUPS
            
            if is_list_default:
                if isinstance(default_val, list):
                    default_options = [x for x in default_val if x in options]
                elif default_val in options:
                    default_options = [default_val]
                else:
                    default_options = []
                    
                st.markdown(f"#### **{group.capitalize()}**")
                selected_val = st.multiselect(f"{group.capitalize()}", options, default=default_options, label_visibility="collapsed", key=f"root_select_{mode}_{group}")
                selected_configs[group] = selected_val
                
                # Subgroups and parameters tree
                if selected_val:
                    with st.expander(f"📁 {group} settings ({', '.join(selected_val)})", expanded=False):
                        for val in selected_val:
                            st.markdown(f"⚙️ **{val} parameters**")
                            st.markdown('<div style="padding-left: 20px; border-left: 2px dashed #666; margin-top: 10px; margin-bottom: 20px;">', unsafe_allow_html=True)
                            render_config_group_ui(group, val, config_overrides, selected_configs)
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Find default index
                default_index = 0
                if default_val in options:
                    default_index = options.index(default_val)
                elif str(default_val).lower() == "null" and "null" in options:
                    default_index = options.index("null")
                elif default_val is None and "None" in options:
                    default_index = options.index("None")
                    
                st.markdown(f"#### **{group.capitalize()}**")
                selected_val = st.selectbox(f"{group.capitalize()}", options, index=default_index, label_visibility="collapsed", key=f"root_select_{mode}_{group}")
                selected_configs[group] = selected_val
                
                # Subgroups and parameters tree
                if selected_val not in ("None", "null", None):
                    with st.expander(f"📁 {group} settings ({selected_val})", expanded=False):
                        render_config_group_ui(group, selected_val, config_overrides, selected_configs)
    
    st.subheader("Other Options")
    seed = st.number_input("Seed (leave blank for null)", value=None, step=1, placeholder="e.g. 42")
    ckpt_path = st.text_input("Checkpoint Path", value="", placeholder="path/to/checkpoint.ckpt")
    
    custom_overrides = st.text_area("Custom Overrides (one per line)", placeholder="trainer.max_epochs=10\ndata.batch_size=32")

# Construct the run command dynamically for real-time display
cmd = srun_prefix + ["uv", "run", script]
for k, v in selected_configs.items():
    if isinstance(v, list):
        if len(v) == 1:
            cmd.append(f"{k}={v[0]}")
        else:
            list_str = f"[{','.join(v)}]"
            cmd.append(f"{k}={list_str}")
    elif v and v != "None" and v != "default" and v != "null":
        cmd.append(f"{k}={v}")
    elif v == "null":
        cmd.append(f"{k}=null")

# Add the edited parameters as overrides
for k, v in config_overrides.items():
    if isinstance(v, bool):
        cmd.append(f"{k}={str(v).lower()}")
    elif isinstance(v, list):
        cmd.append(f"{k}={str(v).replace(' ', '')}")
    else:
        cmd.append(f"{k}={v}")

if seed is not None:
    cmd.append(f"seed={int(seed)}")
if ckpt_path:
    cmd.append(f"ckpt_path={ckpt_path}")
    
if custom_overrides:
    for line in custom_overrides.split("\n"):
        if line.strip():
            cmd.append(line.strip())

with col2:
    st.header("Execution")
    
    if "process" not in st.session_state:
        st.session_state.process = None
        st.session_state.logs = []
        
    run_col, stop_col = st.columns(2)
    with run_col:
        run_btn = st.button("Run", use_container_width=True, type="primary")
    with stop_col:
        stop_btn = st.button("Stop", use_container_width=True)
        
    st.subheader("Generated Command")
    # Merge 'srun ... uv run' and script name into a single line for better display formatting
    display_cmd = []
    if env == "Slurm Cluster":
        srun_len = len(srun_prefix)
        srun_str = " ".join(cmd[:srun_len])
        display_cmd.append(f"{srun_str} uv run {cmd[srun_len + 2]}")
        display_cmd.extend(cmd[srun_len + 3:])
    else:
        if len(cmd) >= 3 and cmd[0] == "uv" and cmd[1] == "run":
            display_cmd.append(f"uv run {cmd[2]}")
            display_cmd.extend(cmd[3:])
        else:
            display_cmd = cmd
            
    st.code(" \\\n  ".join(display_cmd), language="bash")
    
    with st.expander("ℹ️ How to Run on a Slurm Compute Node (via sinteractive)"):
        st.markdown("""
        If you want to run the computation or the GUI on a Slurm compute node rather than the login node:
        
        1. **Start interactive session** on the cluster:
           ```bash
           sinteractive
           ```
           *Enter your resource requirements when prompted.*
        2. **Identify your assigned compute node** from the command prompt (e.g. `username@node042:...` indicates `node042`).
        3. **Launch the Streamlit app** on the compute node:
           ```bash
           uv run streamlit run src/app.py
           ```
        4. **Establish the SSH Tunnel from your local machine** (open a new local terminal):
           ```bash
           ssh -L 8501:node042:8501 username@login.cluster.edu
           ```
           *(Replace `node042` with your actual compute node name and `login.cluster.edu` with your normal login node domain).*
        5. **Access the GUI** in your local browser at `http://localhost:8501`.
        """)
    
    st.subheader("Terminal Output")
    output_container = st.container(height=400)
    with output_container:
        log_area = st.empty()
    
    if stop_btn and st.session_state.process is not None:
        st.session_state.process.terminate()
        st.warning("Process terminated by user.")
        st.session_state.process = None
        
    if run_btn:
        if st.session_state.process is not None:
            st.warning("A process is already running. Stop it first.")
        else:
            st.info(f"Executing: `{' '.join(cmd)}`")
            
            # Start process
            st.session_state.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            st.session_state.logs = []
            
    # Read output
    if st.session_state.process is not None:
        log_output = log_area.code("Running...", language="bash")
        while True:
            # Check if process has finished
            retcode = st.session_state.process.poll()
            
            line = st.session_state.process.stdout.readline()
            if not line and retcode is not None:
                break
                
            if line:
                st.session_state.logs.append(line.strip())
                # keep last 1000 lines
                if len(st.session_state.logs) > 1000:
                    st.session_state.logs = st.session_state.logs[-1000:]
                log_output.code("\n".join(st.session_state.logs), language="bash")
                
        if retcode == 0:
            st.success("Process completed successfully.")
        elif retcode is not None and retcode < 0:
             st.warning(f"Process was terminated (signal {-retcode}).")
        else:
            st.error(f"Process failed with exit code {retcode}")
            
        st.session_state.process = None
    elif st.session_state.logs:
        # Display old logs if process is not running
        log_area.code("\n".join(st.session_state.logs), language="bash")
