import streamlit as st
import importlib
import inspect
import sys
import os

st.set_page_config(page_title="Survey Formula Execution App", layout="wide")
st.title(" Universal Survey Formula Calculator")

# Load all user-uploaded modules
MODULE_PATH = "D:/GitHub/Sampling_design/sampling_log"
if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)

modules = [
    "week1", "week2", "week3", "week4", "week5",
    "week6", "week7", "week10"  # Note: week10 = Week 8 missing data
]

function_map = {}
for mod_name in modules:
    try:
        mod = importlib.import_module(mod_name)
        funcs = {
            name: fn for name, fn in inspect.getmembers(mod, inspect.isfunction)
            if not name.startswith("_")
        }
        function_map.update(funcs)
    except Exception as e:
        st.warning(f"Module load failed: {mod_name} ({e})")

# UI: Select a function
function_list = sorted(function_map.keys())
function_choice = st.selectbox("Select a process/function to execute", function_list)

# Get function object
selected_func = function_map[function_choice]
sig = inspect.signature(selected_func)

# Collect arguments dynamically
st.subheader(f"Inputs for: `{function_choice}`")
# Display concept and parameter info from docstring
docstring = selected_func.__doc__
if docstring:
    st.markdown("###  Description")
    for line in docstring.strip().split("\n"):
        if "Concept:" in line:
            st.info(f" {line.strip()}")
        elif "Parameters:" in line:
            st.markdown(f"**{line.strip()}**")
        elif "-" in line:
            st.markdown(f"- {line.strip()}")

args = {}
for param in sig.parameters.values():
    label = f"{param.name} ({param.annotation.__name__ if param.annotation != inspect._empty else 'value'})"
    default = param.default if param.default != inspect._empty else 0.0
    # Choose input type
    if param.annotation == list or isinstance(default, list):
        raw_input = st.text_input(f"Enter list for `{param.name}` (comma-separated)", "1,2,3")
        try:
            args[param.name] = [float(x.strip()) for x in raw_input.split(",") if x.strip()]
        except:
            st.error(f"Invalid list input for {param.name}")
    else:
        args[param.name] = st.number_input(label, value=float(default) if isinstance(default, (int, float)) else 0.0)

# Compute result
if st.button("Compute"):
    try:
        result = selected_func(**args)
        st.success(f" Output: {result}")
    except Exception as e:
        st.error(f" Error while computing: {e}")
