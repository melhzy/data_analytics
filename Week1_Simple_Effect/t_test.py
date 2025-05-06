# T-Test Visualizer with Fixed Initialization
# Educational tool for understanding statistical significance in Alzheimer's research
import pygame
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Required for pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from scipy import stats
import os
import sys
import traceback

# Initialize pygame
pygame.init()

# Colors for pygame (RGB format)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 180)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
LIGHT_RED = (255, 182, 193)
LIGHT_GREEN = (144, 238, 144)
YELLOW = (255, 255, 0)
DARK_GREEN = (0, 100, 0)
DARK_GRAY = (80, 80, 80)
LIGHT_GRAY = (230, 230, 230)
MEDIUM_GRAY = (180, 180, 180)
PANEL_BLUE = (240, 248, 255)  # AliceBlue
PANEL_GREEN = (240, 255, 240)  # HoneyDew
PANEL_RED = (255, 240, 240)    # MistyRose
PANEL_YELLOW = (255, 255, 224) # LightYellow
TRANSPARENT_WHITE = (255, 255, 255, 180)  # Semi-transparent white

# Colors for matplotlib (normalized RGB format)
MPL_LIGHT_BLUE = (173/255, 216/255, 230/255)
MPL_LIGHT_RED = (255/255, 182/255, 193/255)

# Layout constants for grid-based UI
HEADER_HEIGHT = 60
TITLE_MARGIN = 10
RESULT_MARGIN = 20
PANEL_MARGIN = 15
H_MARGIN = 20
V_MARGIN = 10
SLIDER_SPACING = 40
ROW_SPACING = 40

# Grid layout constants for parameter adjustment section
LABEL_COLUMN_WIDTH = 140   # Fixed width for labels
VALUE_COLUMN_WIDTH = 50    # Fixed width for values
SLIDER_MIN_WIDTH = 150     # Minimum width for sliders
LABEL_SLIDER_GAP = 20      # Gap between label and slider
SLIDER_VALUE_GAP = 16      # Gap between slider and value
SAMPLE_LABEL_WIDTH = 180   # Wider for sample size label

# Screen dimensions - with minimum sizes to prevent overlap
MIN_WIDTH, MIN_HEIGHT = 1000, 700
initial_width, initial_height = 1200, 800
WIDTH, HEIGHT = initial_width, initial_height
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("T-Test Visualizer: Understanding Statistical Significance")

# Fonts - consistent sizing
font_small = pygame.font.SysFont('Arial', 16)
font_medium = pygame.font.SysFont('Arial', 20)
font_large = pygame.font.SysFont('Arial', 24)
font_title = pygame.font.SysFont('Arial', 40, bold=True)
font_result = pygame.font.SysFont('Arial', 22, bold=True)
font_panel_title = pygame.font.SysFont('Arial', 22, bold=True)

# Game states
MENU = 0
GAME = 1
TUTORIAL = 2
RESULTS = 3

# Initial state
state = MENU

# Default sample parameters
sample_size = 20
mean1, std1 = 100, 15  # Control group
mean2, std2 = 115, 15  # Experiment group (initially different)

# Initialize sliders with default values
slider_mean1 = {"pos": (0, 0), "width": 200, "value": mean1, "min": 70, "max": 130, "dragging": False}
slider_std1 = {"pos": (0, 0), "width": 200, "value": std1, "min": 5, "max": 25, "dragging": False}
slider_mean2 = {"pos": (0, 0), "width": 200, "value": mean2, "min": 70, "max": 130, "dragging": False}
slider_std2 = {"pos": (0, 0), "width": 200, "value": std2, "min": 5, "max": 25, "dragging": False}
slider_sample = {"pos": (0, 0), "width": 250, "value": sample_size, "min": 5, "max": 20000, "dragging": False}

# Initialize buttons with empty Rects
calc_button = pygame.Rect(0, 0, 0, 0)
reset_button = pygame.Rect(0, 0, 0, 0)

# Generate initial data
np.random.seed(42)  # For reproducibility
group1 = np.random.normal(mean1, std1, sample_size)
group2 = np.random.normal(mean2, std2, sample_size)

# T-test results
t_stat = 0
p_value = 0
has_calculated = False

# Tutorial pages
tutorial_page = 0
num_tutorial_pages = 5

# Toggleable instruction section
show_instructions = True

# Function to calculate layout parameters based on current window size
def calculate_layout():
    """Calculate all UI component positions based on current window size with proper spacing and no overlaps"""
    global slider_mean1, slider_std1, slider_mean2, slider_std2, slider_sample
    global calc_button, reset_button
    
    # Calculate component heights
    header_height = HEADER_HEIGHT
    results_panel_height = 40
    chart_ratio = 0.4  # Proportion of usable height for chart
    instructions_height = 40
    panel_height = 120
    sample_panel_height = 50
    button_height = 40
    
    # Calculate total available height after header
    available_height = HEIGHT - header_height - PANEL_MARGIN
    
    # Results panel - positioned below title with margin
    results_panel_top = header_height + RESULT_MARGIN
    
    # Calculate remaining height after results panel
    remaining_height = available_height - results_panel_height - RESULT_MARGIN
    
    # Chart dimensions - responsive to window size
    chart_height = min(remaining_height * chart_ratio, 450)
    chart_width = min(WIDTH * 0.6, 700)
    
    # Chart position - centered horizontally, below results panel
    chart_left = (WIDTH - chart_width) // 2
    chart_top = results_panel_top + results_panel_height + PANEL_MARGIN
    
    # Instructions panel - placed below the chart
    instructions_top = chart_top + chart_height + PANEL_MARGIN
    
    # Calculate remaining height for panels and buttons
    remaining_height_after_chart = HEIGHT - instructions_top - instructions_height - PANEL_MARGIN
    
    # Determine if we need to reduce panel heights to fit everything
    total_needed_height = panel_height + PANEL_MARGIN + sample_panel_height + PANEL_MARGIN*3 + button_height
    
    # Adjust panel heights if needed to fit everything
    if total_needed_height > remaining_height_after_chart:
        # Calculate scaling factor
        scale_factor = remaining_height_after_chart / total_needed_height
        # Scale panel heights while ensuring minimum sizes
        panel_height = max(80, int(panel_height * scale_factor))
        sample_panel_height = max(40, int(sample_panel_height * scale_factor))
    
    # Panel dimensions - responsive to window size with minimum widths
    min_panel_width = LABEL_COLUMN_WIDTH + SLIDER_MIN_WIDTH + VALUE_COLUMN_WIDTH + LABEL_SLIDER_GAP + SLIDER_VALUE_GAP + H_MARGIN*2
    panel_width = max(min(WIDTH * 0.4, 450), min_panel_width)
    
    # Panel positions with better distribution
    left_panel_left = max((WIDTH - 2 * panel_width - PANEL_MARGIN) // 2, H_MARGIN)
    right_panel_left = left_panel_left + panel_width + PANEL_MARGIN
    panel_top = instructions_top + instructions_height + PANEL_MARGIN
    
    # Define grid for each panel with proper column layout and improved vertical spacing
    title_height = 40  # Space for the title at the top of each panel
    
    # Calculate available space for sliders after accounting for title
    available_slider_space = panel_height - title_height
    
    # Distribute slider rows evenly in the available space
    control_grid = {
        'label_col': left_panel_left + H_MARGIN,
        'label_width': LABEL_COLUMN_WIDTH,
        'slider_col': left_panel_left + H_MARGIN + LABEL_COLUMN_WIDTH + LABEL_SLIDER_GAP,
        'slider_width': panel_width - LABEL_COLUMN_WIDTH - VALUE_COLUMN_WIDTH - H_MARGIN*2 - LABEL_SLIDER_GAP - SLIDER_VALUE_GAP,
        'value_col': left_panel_left + panel_width - VALUE_COLUMN_WIDTH - H_MARGIN,
        'value_width': VALUE_COLUMN_WIDTH,
        'row1': panel_top + title_height + available_slider_space * 0.25,  # First slider at 1/4 of available space
        'row2': panel_top + title_height + available_slider_space * 0.75   # Second slider at 3/4 of available space
    }
    
    # Similar updates for experimental panel grid
    exp_grid = {
        'label_col': right_panel_left + H_MARGIN,
        'label_width': LABEL_COLUMN_WIDTH,
        'slider_col': right_panel_left + H_MARGIN + LABEL_COLUMN_WIDTH + LABEL_SLIDER_GAP,
        'slider_width': panel_width - LABEL_COLUMN_WIDTH - VALUE_COLUMN_WIDTH - H_MARGIN*2 - LABEL_SLIDER_GAP - SLIDER_VALUE_GAP,
        'value_col': right_panel_left + panel_width - VALUE_COLUMN_WIDTH - H_MARGIN,
        'value_width': VALUE_COLUMN_WIDTH,
        'row1': panel_top + title_height + available_slider_space * 0.25,  # First slider at 1/4 of available space
        'row2': panel_top + title_height + available_slider_space * 0.75   # Second slider at 3/4 of available space
    }
    
    # Sample size panel - centered below group panels with fixed vertical spacing
    sample_panel_width = min(WIDTH * 0.6, 550)
    sample_panel_left = (WIDTH - sample_panel_width) // 2
    sample_panel_top = panel_top + panel_height + PANEL_MARGIN
    
    # Ensure sample panel doesn't overlap with buttons
    available_button_space = HEIGHT - (sample_panel_top + sample_panel_height) - PANEL_MARGIN*3
    
    # Sample size grid
    sample_grid = {
        'label_col': sample_panel_left + H_MARGIN,
        'label_width': SAMPLE_LABEL_WIDTH,
        'slider_col': sample_panel_left + H_MARGIN + SAMPLE_LABEL_WIDTH + LABEL_SLIDER_GAP,
        'slider_width': sample_panel_width - SAMPLE_LABEL_WIDTH - VALUE_COLUMN_WIDTH - H_MARGIN*2 - LABEL_SLIDER_GAP - SLIDER_VALUE_GAP,
        'value_col': sample_panel_left + sample_panel_width - VALUE_COLUMN_WIDTH - H_MARGIN,
        'value_width': VALUE_COLUMN_WIDTH,
        'row': sample_panel_top + sample_panel_height // 2  # Center vertically
    }
    
    # Button positions - ensure they're properly positioned below sample panel
    button_width = 200
    button_height = min(40, available_button_space)  # Ensure buttons fit
    button_spacing = 20
    
    # Ensure there's vertical space for buttons
    button_top = sample_panel_top + sample_panel_height + PANEL_MARGIN * 3
    
    # If buttons would go off screen, adjust their position
    if button_top + button_height + PANEL_MARGIN > HEIGHT:
        button_top = HEIGHT - button_height - PANEL_MARGIN
    
    # Calculate horizontal button positions
    left_button_left = (WIDTH - 2 * button_width - button_spacing) // 2
    right_button_left = left_button_left + button_width + button_spacing
    
    # Update slider positions using the improved grid
    slider_mean1 = {
        "pos": (control_grid['slider_col'], control_grid['row1']),
        "width": control_grid['slider_width'],
        "value": slider_mean1["value"],
        "min": 70,
        "max": 130,
        "dragging": slider_mean1["dragging"]
    }
    
    slider_std1 = {
        "pos": (control_grid['slider_col'], control_grid['row2']),
        "width": control_grid['slider_width'],
        "value": slider_std1["value"],
        "min": 5,
        "max": 25,
        "dragging": slider_std1["dragging"]
    }
    
    # Experimental group sliders
    slider_mean2 = {
        "pos": (exp_grid['slider_col'], exp_grid['row1']),
        "width": exp_grid['slider_width'],
        "value": slider_mean2["value"],
        "min": 70,
        "max": 130,
        "dragging": slider_mean2["dragging"]
    }
    
    slider_std2 = {
        "pos": (exp_grid['slider_col'], exp_grid['row2']),
        "width": exp_grid['slider_width'],
        "value": slider_std2["value"],
        "min": 5,
        "max": 25,
        "dragging": slider_std2["dragging"]
    }
    
    # Sample size slider
    slider_sample = {
        "pos": (sample_grid['slider_col'], sample_grid['row']),
        "width": sample_grid['slider_width'],
        "value": slider_sample["value"],
        "min": 5,
        "max": 20000,
        "dragging": slider_sample["dragging"]
    }
    
    # Update buttons
    calc_button = pygame.Rect(left_button_left, button_top, button_width, button_height)
    reset_button = pygame.Rect(right_button_left, button_top, button_width, button_height)
    
    # Return all layout parameters for use in drawing functions
    return {
        "header_height": header_height,
        "chart": {
            "left": chart_left,
            "top": chart_top,
            "width": chart_width,
            "height": chart_height
        },
        "results": {
            "left": chart_left,
            "top": results_panel_top,
            "width": chart_width,
            "height": results_panel_height
        },
        "instructions": {
            "left": left_panel_left,
            "top": instructions_top,
            "width": WIDTH - 2 * left_panel_left,
            "height": instructions_height
        },
        "control_panel": {
            "left": left_panel_left,
            "top": panel_top,
            "width": panel_width,
            "height": panel_height,
            "grid": control_grid
        },
        "exp_panel": {
            "left": right_panel_left,
            "top": panel_top,
            "width": panel_width,
            "height": panel_height,
            "grid": exp_grid
        },
        "sample_panel": {
            "left": sample_panel_left,
            "top": sample_panel_top,
            "width": sample_panel_width,
            "height": sample_panel_height,
            "grid": sample_grid
        },
        "toggle_rect": pygame.Rect(
            left_panel_left, 
            instructions_top, 
            30, 30
        ),
        "buttons": {
            "top": button_top,
            "height": button_height
        }
    }

def debug_info(message):
    """Print debug information"""
    print(f"DEBUG: {message}")

def draw_panel(rect, color, border_radius=10, border_color=None, border_width=0):
    """Draw a panel with optional border"""
    pygame.draw.rect(screen, color, rect, border_radius=border_radius)
    if border_color:
        pygame.draw.rect(screen, border_color, rect, width=border_width, border_radius=border_radius)

def draw_text(text, font, color, x, y, align="left"):
    """Draw text on screen with alignment options"""
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    
    if align == "center":
        text_rect.center = (x, y)
    elif align == "right":
        text_rect.right = x
        text_rect.centery = y
    else:  # left align
        text_rect.left = x
        text_rect.centery = y
        
    screen.blit(text_surface, text_rect)
    return text_rect

def draw_slider(slider, label, grid_pos, mouse_pos, value_format="{:.1f}", value_suffix=""):
    """Draw a slider with label and current value using grid layout with hover effects"""
    # Create hover area for slider track
    slider_rect = pygame.Rect(slider["pos"][0] - 5, slider["pos"][1] - 10, slider["width"] + 10, 20)
    
    # Check if mouse is hovering over this slider
    is_hovering = slider_rect.collidepoint(mouse_pos)
    
    # Draw slider track with different color when hovered
    track_color = BLUE if is_hovering or slider["dragging"] else GRAY
    track_alpha = 200 if is_hovering or slider["dragging"] else 150
    
    pygame.draw.rect(screen, track_color, (slider["pos"][0], slider["pos"][1] - 5, slider["width"], 10))
    
    # Calculate handle position
    handle_pos = slider["pos"][0] + (slider["value"] - slider["min"]) / (slider["max"] - slider["min"]) * slider["width"]
    
    # Draw handle - larger when hovered or dragging
    handle_radius = 12 if is_hovering or slider["dragging"] else 10
    handle_color = DARK_BLUE if is_hovering or slider["dragging"] else BLUE
    pygame.draw.circle(screen, handle_color, (int(handle_pos), slider["pos"][1]), handle_radius)
    
    # Draw label with fixed position and right alignment
    label_x = slider["pos"][0] - LABEL_SLIDER_GAP
    draw_text(label, font_medium, BLACK, label_x, slider["pos"][1], "right")
    
    # Draw value with fixed position and left alignment
    value_x = slider["pos"][0] + slider["width"] + SLIDER_VALUE_GAP
    draw_text(value_format.format(slider["value"]) + value_suffix, font_medium, BLACK, value_x, slider["pos"][1], "left")
    
    return is_hovering

def add_hover_tooltip(rect, text, mouse_pos):
    """Show tooltip when hovering over an element"""
    # Check if rect is a tuple of (x, y) coordinates or a Rect object
    if isinstance(rect, tuple):
        # Create a small Rect around the position
        x, y = rect
        hover_area = pygame.Rect(x - 10, y - 10, 20, 20)
        is_hovering = hover_area.collidepoint(mouse_pos)
    else:
        # Use the rect directly if it's already a Rect object
        is_hovering = rect.collidepoint(mouse_pos)
        
    if is_hovering:
        tooltip_surface = font_small.render(text, True, BLACK, LIGHT_GRAY)
        tooltip_rect = tooltip_surface.get_rect(topleft=(mouse_pos[0] + 10, mouse_pos[1] + 10))
        
        # Ensure tooltip stays within screen bounds
        if tooltip_rect.right > WIDTH:
            tooltip_rect.right = WIDTH - 5
        if tooltip_rect.bottom > HEIGHT:
            tooltip_rect.bottom = HEIGHT - 5
            
        # Draw tooltip background
        pygame.draw.rect(screen, LIGHT_GRAY, tooltip_rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, tooltip_rect, width=1, border_radius=5)
        
        # Draw tooltip text
        screen.blit(tooltip_surface, tooltip_rect)
        return True
    return False

def create_histogram_surface(group1, group2, width=600, height=400):
    """Create a matplotlib histogram figure with improved layout and padding"""
    try:
        debug_info(f"Creating histogram with: group1 size={len(group1)}, mean={np.mean(group1):.2f}, std={np.std(group1):.2f}")
        debug_info(f"Creating histogram with: group2 size={len(group2)}, mean={np.mean(group2):.2f}, std={np.std(group2):.2f}")
        
        # Handle empty or invalid data
        if len(group1) < 2 or len(group2) < 2:
            raise ValueError(f"Not enough data points: group1={len(group1)}, group2={len(group2)} - need at least 2 in each")
            
        # Create figure with proper scaling
        dpi = 100
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        
        # Add more space at the bottom for the x-axis label
        fig.subplots_adjust(top=0.85, right=0.85, bottom=0.15)  # Added bottom padding
        
        ax = fig.add_subplot(111)
        
        # Create bins that cover both groups with extra padding
        all_data = np.concatenate([group1, group2])
        if len(all_data) < 2 or (np.max(all_data) - np.min(all_data) < 0.001):
            # If all values are identical or nearly so, create artificial bins
            mean_val = np.mean(all_data)
            bins = np.linspace(mean_val - 10, mean_val + 10, 20)
            debug_info(f"Using artificial bins: min={bins[0]}, max={bins[-1]}")
        else:
            min_val = np.min(all_data)
            max_val = np.max(all_data)
            # Ensure bins have some padding
            range_val = max_val - min_val
            min_val -= range_val * 0.2  # Increased padding
            max_val += range_val * 0.2
            
            # Dynamic number of bins based on chart width
            num_bins = max(10, min(20, int(width/40)))  # Scale bins with width
            bins = np.linspace(min_val, max_val, num_bins)
            debug_info(f"Using data-based bins: min={bins[0]}, max={bins[-1]}, num_bins={num_bins}")
        
        # Plot histograms with better alpha for visibility
        ax.hist(group1, bins=bins, alpha=0.6, label='Control Group', color=MPL_LIGHT_BLUE, edgecolor='blue', linewidth=0.5)
        ax.hist(group2, bins=bins, alpha=0.6, label='Experimental Group', color=MPL_LIGHT_RED, edgecolor='red', linewidth=0.5)
        
        # Add vertical lines for means
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        ax.axvline(mean1, color='blue', linestyle='dashed', linewidth=2, label=f'Mean 1: {mean1:.1f}')
        ax.axvline(mean2, color='red', linestyle='dashed', linewidth=2, label=f'Mean 2: {mean2:.1f}')
        
        # Make legend more compact with smaller font size
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        
        # Set axis labels with increased font size for better visibility
        # ax.set_xlabel('Value', fontsize=12, labelpad=10)  # Increased font size and padding
        ax.set_ylabel('Frequency', fontsize=12)
        
        # Make sure x-axis limits are appropriate with additional padding
        current_xlim = ax.get_xlim()
        padding = (current_xlim[1] - current_xlim[0]) * 0.1  # 10% padding
        ax.set_xlim(current_xlim[0] - padding, current_xlim[1] + padding)
        
        # Convert matplotlib figure to pygame surface
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        plt.close(fig)
        
        surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
        return surf
        
    except Exception as e:
        print(f"Error creating histogram: {e}")
        print(traceback.format_exc())
        
        # Create empty surface with error message
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        
        error_message = f"Error creating histogram.\nTry different parameters."
        ax.text(0.5, 0.5, error_message, ha='center', va='center', color='red', fontsize=14)
        ax.axis('off')
        
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        plt.close(fig)
        
        surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
        return surf
    
def draw_tutorial():
    """Draw tutorial screens based on current page"""
    screen.fill(WHITE)
    
    # Header
    draw_text("T-Test Tutorial", font_title, BLUE, WIDTH // 2, 50, "center")
    
    if tutorial_page == 0:
        # Introduction
        draw_text("What is a t-test?", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "A t-test is a statistical test used to determine if there is a significant",
            "difference between the means of two groups.",
            "",
            "It's widely used in medical research, including Alzheimer's studies, to compare:",
            "• Control vs. treatment groups",
            "• Before vs. after treatment",
            "• Different patient populations",
            "",
            "This interactive simulation will help you understand how sample size, means,",
            "and variability affect statistical significance."
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    elif tutorial_page == 1:
        # Sample means
        draw_text("Sample Means", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "The t-test compares the means of two groups.",
            "",
            "In the simulation:",
            "• Blue points represent the control group (e.g., healthy subjects)",
            "• Red points represent the experimental group (e.g., Alzheimer's patients)",
            "",
            "You can adjust the mean of each group using sliders.",
            "When the means are very different, it's easier to detect a significant difference.",
            "",
            "Try to find out: How far apart do the means need to be for significance?"
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    elif tutorial_page == 2:
        # Standard deviation
        draw_text("Variability (Standard Deviation)", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "The standard deviation measures the spread of data points.",
            "",
            "Higher standard deviation means:",
            "• More variability in your measurements",
            "• Points are more scattered",
            "• It's harder to detect true differences between groups",
            "",
            "In Alzheimer's research, high variability might come from:",
            "• Different disease stages",
            "• Genetic differences",
            "• Measurement errors",
            "",
            "Try changing the standard deviation to see how it affects significance."
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    elif tutorial_page == 3:
        # Sample size
        draw_text("Sample Size", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "Sample size refers to how many subjects are in each group.",
            "",
            "Larger sample sizes:",
            "• Increase statistical power",
            "• Make it easier to detect true differences",
            "• Reduce the impact of outliers",
            "",
            "In clinical trials for Alzheimer's, researchers must carefully choose",
            "sample sizes that balance statistical power with practical limitations.",
            "",
            "Try increasing the sample size to see how it affects your ability",
            "to detect differences between groups."
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    elif tutorial_page == 4:
        # P-value explanation
        draw_text("Understanding p-values", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "The p-value tells you the probability that the observed difference",
            "between groups occurred by random chance.",
            "",
            "• p < 0.05: Statistically significant (95% confidence)",
            "• p < 0.01: Highly significant (99% confidence)",
            "• p < 0.001: Very highly significant (99.9% confidence)",
            "",
            "In Alzheimer's research, a significant p-value might indicate:",
            "• A treatment has a real effect",
            "• A biomarker is truly different between groups",
            "• A genetic variant is associated with the disease",
            "",
            "Remember: Statistical significance doesn't always mean clinical significance!"
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    # Navigation buttons
    back_button = pygame.Rect(50, HEIGHT - 100, 200, 50)
    next_button = pygame.Rect(WIDTH - 250, HEIGHT - 100, 200, 50)
    
    pygame.draw.rect(screen, LIGHT_BLUE, back_button, border_radius=5)
    pygame.draw.rect(screen, LIGHT_BLUE, next_button, border_radius=5)
    
    if tutorial_page > 0:
        draw_text("Previous", font_medium, BLACK, back_button.centerx, back_button.centery, "center")
    
    if tutorial_page < num_tutorial_pages - 1:
        draw_text("Next", font_medium, BLACK, next_button.centerx, next_button.centery, "center")
    else:
        draw_text("Start Simulation", font_medium, BLACK, next_button.centerx, next_button.centery, "center")
    
    # Page indicator
    draw_text(f"Page {tutorial_page + 1}/{num_tutorial_pages}", font_small, BLACK, WIDTH // 2, HEIGHT - 75, "center")

def draw_instructions_toggle(toggle_rect):
    """Draw a toggle button for instructions"""
    pygame.draw.rect(screen, LIGHT_BLUE if show_instructions else LIGHT_GRAY, toggle_rect)
    
    # Draw icon for info
    pygame.draw.rect(screen, BLACK, toggle_rect, width=2)
    draw_text("i", font_medium, BLACK, toggle_rect.centerx, toggle_rect.centery, "center")
    
    return toggle_rect

def draw_menu():
    """Draw the main menu screen"""
    screen.fill(WHITE)
    
    # Draw title
    title_rect = draw_text("T-Test Visualizer", font_title, BLUE, WIDTH // 2, HEIGHT // 4, "center")
    
    # Draw subtitle
    subtitle_rect = draw_text("A Visual Exploration of Statistical Significance", font_large, 
                             BLACK, WIDTH // 2, title_rect.bottom + 30, "center")
    
    # Draw description
    description = [
        "Understand how t-tests work in clinical research",
        "Perfect for Alzheimer's disease studies and statistical analysis",
        "Explore the effects of sample size, means, and variability"
    ]
    
    for i, line in enumerate(description):
        draw_text(line, font_medium, BLACK, WIDTH // 2, subtitle_rect.bottom + 50 + i * 40, "center")
    
    # Draw buttons
    start_button = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 100, 300, 60)
    tutorial_button = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 200, 300, 60)
    
    pygame.draw.rect(screen, LIGHT_BLUE, start_button, border_radius=10)
    pygame.draw.rect(screen, LIGHT_GREEN, tutorial_button, border_radius=10)
    
    draw_text("Start Simulation", font_large, BLACK, start_button.centerx, start_button.centery, "center")
    draw_text("Tutorial", font_large, BLACK, tutorial_button.centerx, tutorial_button.centery, "center")
    
    # Footer
    draw_text("Created for Alzheimer's Research Statistical Education", font_small, GRAY, 
             WIDTH // 2, HEIGHT - 50, "center")
    
    return start_button, tutorial_button

def draw_game(layout, mouse_pos):
    """Draw the main game screen with improved responsive layout"""
    screen.fill(WHITE)
    
    # Draw header with clear separation
    header_bg = pygame.Rect(0, 0, WIDTH, layout["header_height"])
    pygame.draw.rect(screen, PANEL_BLUE, header_bg)
    draw_text("T-Test Visualizer", font_title, BLUE, WIDTH // 2, layout["header_height"] // 2, "center")
    
    # Draw back button
    back_button = pygame.Rect(30, 10, 100, 40)
    pygame.draw.rect(screen, LIGHT_RED, back_button, border_radius=5)
    draw_text("Menu", font_small, BLACK, back_button.centerx, back_button.centery, "center")
    
    # Draw results panel - COMPLETELY SEPARATE from title
    if has_calculated:
        result_panel_color = PANEL_GREEN if p_value < 0.05 else PANEL_RED
        result_text_color = DARK_GREEN if p_value < 0.05 else RED
        result_text = "Significant Difference!" if p_value < 0.05 else "No Significant Difference"
        
        # Results panel with clear separation from histogram
        result_panel = pygame.Rect(
            layout["results"]["left"], 
            layout["results"]["top"], 
            layout["results"]["width"], 
            layout["results"]["height"]
        )
        draw_panel(result_panel, result_panel_color, border_radius=5, border_color=DARK_GRAY, border_width=1)
        
        # Draw results with proper spacing and division into thirds
        panel_width = layout["results"]["width"]
        t_pos_x = result_panel.left + 10
        draw_text(f"t-statistic: {t_stat:.3f}", font_result, BLACK, t_pos_x, result_panel.centery)
        
        p_pos_x = result_panel.left + panel_width // 3
        draw_text(f"p-value: {p_value:.4f}", font_result, BLACK, p_pos_x, result_panel.centery)
        
        # Draw significance result - right aligned
        draw_text(result_text, font_result, result_text_color, result_panel.right - 10, result_panel.centery, "right")
    
    # Create and draw histogram in its own panel with proper padding
    chart_rect = pygame.Rect(
        layout["chart"]["left"], 
        layout["chart"]["top"], 
        layout["chart"]["width"], 
        layout["chart"]["height"]
    )
    draw_panel(chart_rect, WHITE, border_color=GRAY, border_width=2)
    
    try:
        hist_surface = create_histogram_surface(
            group1, group2, 
            width=layout["chart"]["width"] - 20, 
            height=layout["chart"]["height"] - 20
        )
        screen.blit(hist_surface, (layout["chart"]["left"] + 10, layout["chart"]["top"] + 10))
    except Exception as e:
        # Fallback if histogram creation fails
        pygame.draw.rect(screen, LIGHT_BLUE, (
            layout["chart"]["left"] + 10, 
            layout["chart"]["top"] + 10, 
            layout["chart"]["width"] - 20, 
            layout["chart"]["height"] - 20
        ))
        draw_text("Error creating histogram. Try adjusting parameters.", 
                 font_medium, RED, WIDTH // 2, layout["chart"]["top"] + layout["chart"]["height"] // 2, "center")
        print(f"Histogram error: {e}")
    
    # Instructions toggle button - moved to top-left of instructions area
    toggle_rect = draw_instructions_toggle(layout["toggle_rect"])
    
    # Draw instructions if enabled
    if show_instructions:
        instruction_panel = pygame.Rect(
            layout["instructions"]["left"], 
            layout["instructions"]["top"], 
            layout["instructions"]["width"], 
            layout["instructions"]["height"]
        )
        draw_panel(instruction_panel, PANEL_YELLOW, border_color=DARK_GRAY, border_width=1)
        instruction_text = "Adjust the sliders to see their impact on the t-test and aim for p < 0.05."
        
        # Center the text in the instruction panel
        draw_text(instruction_text, font_medium, BLACK, instruction_panel.centerx, instruction_panel.centery, "center")
        
    # Draw control group panel with grid layout
    control_panel = pygame.Rect(
        layout["control_panel"]["left"], 
        layout["control_panel"]["top"], 
        layout["control_panel"]["width"], 
        layout["control_panel"]["height"]
    )
    draw_panel(control_panel, PANEL_BLUE, border_color=DARK_GRAY, border_width=1)

    # Control group title - centered horizontally at the top of the panel
    title_y = control_panel.top + 20
    draw_text("Control Group", font_panel_title, BLACK, control_panel.centerx, title_y, "center")
    
    # Draw experimental group panel - exactly aligned with control panel
    exp_panel = pygame.Rect(
        layout["exp_panel"]["left"], 
        layout["exp_panel"]["top"], 
        layout["exp_panel"]["width"], 
        layout["exp_panel"]["height"]
    )
    draw_panel(exp_panel, PANEL_RED, border_color=DARK_GRAY, border_width=1)
    
    # Experimental group title - centered horizontally at the top of the panel
    draw_text("Experimental Group", font_panel_title, BLACK, exp_panel.centerx, title_y, "center")
    
    # Draw sample size panel with consistent styling
    sample_panel = pygame.Rect(
        layout["sample_panel"]["left"], 
        layout["sample_panel"]["top"], 
        layout["sample_panel"]["width"], 
        layout["sample_panel"]["height"]
    )
    draw_panel(sample_panel, LIGHT_GRAY, border_color=DARK_GRAY, border_width=1)
    
    # Draw all sliders with improved hover detection and grid layout
    mean1_hover = draw_slider(slider_mean1, "Mean:", layout["control_panel"]["grid"], mouse_pos, "{:.1f}")
    std1_hover = draw_slider(slider_std1, "Standard Deviation:", layout["control_panel"]["grid"], mouse_pos, "{:.1f}")
    
    mean2_hover = draw_slider(slider_mean2, "Mean:", layout["exp_panel"]["grid"], mouse_pos, "{:.1f}")
    std2_hover = draw_slider(slider_std2, "Standard Deviation:", layout["exp_panel"]["grid"], mouse_pos, "{:.1f}")
    
    # Draw sample size slider
    sample_hover = draw_slider(slider_sample, "Sample Size (2 groups):", layout["sample_panel"]["grid"], mouse_pos, "{:.0f}")
    
    # Draw buttons
    pygame.draw.rect(screen, GREEN if not has_calculated else GRAY, calc_button, border_radius=5)
    draw_text("Calculate T-Test", font_medium, BLACK, calc_button.centerx, calc_button.centery, "center")
    
    pygame.draw.rect(screen, LIGHT_BLUE, reset_button, border_radius=5)
    draw_text("Reset Data", font_medium, BLACK, reset_button.centerx, reset_button.centery, "center")
    
    # Add tooltips for interactive elements when hovering
    if (not mean1_hover and not std1_hover and not mean2_hover and not std2_hover and not sample_hover):
        # Only show tooltips if we're not already hovering over a slider
        handle_size = 20
        mean1_rect = pygame.Rect(slider_mean1["pos"][0] - handle_size//2, slider_mean1["pos"][1] - handle_size//2, handle_size, handle_size)
        std1_rect = pygame.Rect(slider_std1["pos"][0] - handle_size//2, slider_std1["pos"][1] - handle_size//2, handle_size, handle_size)
        mean2_rect = pygame.Rect(slider_mean2["pos"][0] - handle_size//2, slider_mean2["pos"][1] - handle_size//2, handle_size, handle_size)
        std2_rect = pygame.Rect(slider_std2["pos"][0] - handle_size//2, slider_std2["pos"][1] - handle_size//2, handle_size, handle_size)
        sample_rect = pygame.Rect(slider_sample["pos"][0] - handle_size//2, slider_sample["pos"][1] - handle_size//2, handle_size, handle_size)
        
        if add_hover_tooltip(mean1_rect, "Adjust the mean value of the control group", mouse_pos):
            pass
        elif add_hover_tooltip(std1_rect, "Adjust the variability of the control group", mouse_pos):
            pass
        elif add_hover_tooltip(mean2_rect, "Adjust the mean value of the experimental group", mouse_pos):
            pass
        elif add_hover_tooltip(std2_rect, "Adjust the variability of the experimental group", mouse_pos):
            pass
        elif add_hover_tooltip(sample_rect, "Adjust the number of subjects in each group", mouse_pos):
            pass
    
    return back_button, toggle_rect

def reset_data():
    """Reset the simulation data based on current slider values with validation"""
    global group1, group2, has_calculated
    
    # Get sample size (ensure at least 5 samples for better visualization)
    sample_size = max(5, int(slider_sample["value"]))
    
    # Make sure standard deviations are positive
    std1 = max(0.1, slider_std1["value"])
    std2 = max(0.1, slider_std2["value"])
    
    debug_info(f"Generating new data: mean1={slider_mean1['value']:.2f}, std1={std1:.2f}, mean2={slider_mean2['value']:.2f}, std2={std2:.2f}, n={sample_size}")
    
    # Generate new data based on slider values
    try:
        group1 = np.random.normal(slider_mean1["value"], std1, sample_size)
        group2 = np.random.normal(slider_mean2["value"], std2, sample_size)
        
        debug_info(f"Generated data: group1 mean={np.mean(group1):.2f}, std={np.std(group1):.2f}, group2 mean={np.mean(group2):.2f}, std={np.std(group2):.2f}")
        
        has_calculated = False
    except Exception as e:
        print(f"Error generating data: {e}")
        print(traceback.format_exc())
        
        # Fallback to simple data if generation fails
        group1 = np.array([slider_mean1["value"]] * sample_size) + np.random.random(sample_size)
        group2 = np.array([slider_mean2["value"]] * sample_size) + np.random.random(sample_size)
        has_calculated = False

def calculate_ttest():
    """Calculate t-test on current data"""
    global t_stat, p_value, has_calculated
    
    try:
        debug_info(f"Calculating t-test: group1 size={len(group1)}, group2 size={len(group2)}")
        
        # Ensure we have enough data for a t-test
        if len(group1) < 2 or len(group2) < 2:
            t_stat = 0
            p_value = 1.0
            debug_info(f"Not enough data for t-test. Using defaults.")
        else:
            # Perform independent t-test
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
            
            # Take absolute value of t-stat for simpler interpretation
            t_stat = abs(t_stat)
            
            # Check for invalid results
            if np.isnan(t_stat) or np.isnan(p_value):
                t_stat = 0
                p_value = 1.0
                debug_info("Warning: NaN values detected in t-test results. Using default values.")
        
        has_calculated = True
        debug_info(f"T-test results: t={t_stat:.4f}, p={p_value:.4f}")
    except Exception as e:
        # Handle any errors during calculation
        t_stat = 0
        p_value = 1.0
        has_calculated = True
        print(f"Error calculating t-test: {e}")
        print(traceback.format_exc())

# Initialize layout at startup
layout = calculate_layout()

# Main game loop
running = True
clock = pygame.time.Clock()

# Recalculate minimum height based on component heights
# Improved calculate_min_height function
def calculate_min_height():
    """Calculate minimum required height for all UI elements"""
    # Base component heights
    header_height = HEADER_HEIGHT
    results_height = 40
    chart_min_height = 300  # Minimum reasonable chart height
    instructions_height = 40
    panel_min_height = 80  # Minimum viable panel height
    sample_panel_min_height = 40
    button_min_height = 40
    
    # Minimum margins between components
    min_margins = PANEL_MARGIN * 5
    
    # Calculate minimum required height
    min_required_height = (header_height + RESULT_MARGIN + 
                          results_height + PANEL_MARGIN + 
                          chart_min_height + PANEL_MARGIN + 
                          instructions_height + PANEL_MARGIN + 
                          panel_min_height + PANEL_MARGIN + 
                          sample_panel_min_height + PANEL_MARGIN*3 + 
                          button_min_height + PANEL_MARGIN)
    
    return max(min_required_height, 700)  # Keep minimum of 700px for aesthetic reasons
    # Estimate the total required height for all components
    header_height = HEADER_HEIGHT
    results_height = 40
    chart_height = min(HEIGHT * 0.45, 450)
    instructions_height = 40
    panel_height = 120
    sample_panel_height = 50
    button_height = 40
    
    # Add margins between components
    total_margins = PANEL_MARGIN * 6
    
    # Calculate minimum required height
    required_height = (header_height + results_height + chart_height + 
                       instructions_height + panel_height + 
                       sample_panel_height + button_height + total_margins)
    
    return max(required_height, 700)  # Keep minimum of 700px

# Update MIN_HEIGHT using this calculation
# MIN_HEIGHT = calculate_min_height()

while running:
    mouse_pos = pygame.mouse.get_pos()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.VIDEORESIZE:
            # Calculate minimum dimensions
            calculated_min_width = 2 * (LABEL_COLUMN_WIDTH + SLIDER_MIN_WIDTH + VALUE_COLUMN_WIDTH) + PANEL_MARGIN*3 + H_MARGIN*4
            calculated_min_height = calculate_min_height()
            
            # Handle window resizing with dynamic minimum dimensions
            WIDTH = max(event.w, calculated_min_width, MIN_WIDTH)
            HEIGHT = max(event.h, calculated_min_height)
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            
            # Recalculate layout with new dimensions
            layout = calculate_layout()
            
            # Prevent any current slider dragging operations from continuing
            for slider in [slider_mean1, slider_mean2, slider_std1, slider_std2, slider_sample]:
                slider["dragging"] = False
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                # Return to menu when escape is pressed
                if state != MENU:
                    state = MENU
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                
                if state == MENU:
                    # Check for button clicks in menu
                    start_button, tutorial_button = draw_menu()
                    
                    if start_button.collidepoint(mouse_pos):
                        state = GAME
                        reset_data()
                    elif tutorial_button.collidepoint(mouse_pos):
                        state = TUTORIAL
                        tutorial_page = 0
                
                elif state == TUTORIAL:
                    # Check for button clicks in tutorial
                    back_button = pygame.Rect(50, HEIGHT - 100, 200, 50)
                    next_button = pygame.Rect(WIDTH - 250, HEIGHT - 100, 200, 50)
                    menu_button = pygame.Rect(30, 10, 100, 40)
                    
                    if menu_button.collidepoint(mouse_pos):
                        state = MENU
                    elif back_button.collidepoint(mouse_pos) and tutorial_page > 0:
                        tutorial_page -= 1
                    elif next_button.collidepoint(mouse_pos):
                        if tutorial_page < num_tutorial_pages - 1:
                            tutorial_page += 1
                        else:
                            state = GAME
                            reset_data()
                
                elif state == GAME:
                    # Get UI elements from the draw function
                    back_button, toggle_rect = draw_game(layout, mouse_pos)
                    
                    # Check for instruction toggle
                    if toggle_rect.collidepoint(mouse_pos):
                        show_instructions = not show_instructions
                        
                    # Check for button clicks in game
                    if back_button.collidepoint(mouse_pos):
                        state = MENU
                    elif calc_button.collidepoint(mouse_pos):
                        calculate_ttest()
                    elif reset_button.collidepoint(mouse_pos):
                        reset_data()
                    
                    # Check for slider interactions with better hit detection
                    for slider in [slider_mean1, slider_mean2, slider_std1, slider_std2, slider_sample]:
                        # Create a wider hit area for the entire slider track
                        slider_rect = pygame.Rect(
                            slider["pos"][0] - 5, 
                            slider["pos"][1] - 10, 
                            slider["width"] + 10, 
                            20
                        )
                        if slider_rect.collidepoint(mouse_pos):
                            slider["dragging"] = True
                            # Update the value immediately based on click position
                            slider["value"] = (mouse_pos[0] - slider["pos"][0]) / slider["width"] * (slider["max"] - slider["min"]) + slider["min"]
                            slider["value"] = max(slider["min"], min(slider["max"], slider["value"]))
        
        elif event.type == pygame.MOUSEBUTTONUP:
            # Stop dragging
            for slider in [slider_mean1, slider_mean2, slider_std1, slider_std2, slider_sample]:
                if slider["dragging"]:
                    slider["dragging"] = False
                    reset_data()  # Reset data when slider is released
        
        elif event.type == pygame.MOUSEMOTION:
            # Update sliders
            for slider in [slider_mean1, slider_mean2, slider_std1, slider_std2, slider_sample]:
                if slider["dragging"]:
                    # Calculate new value based on mouse position
                    new_val = (mouse_pos[0] - slider["pos"][0]) / slider["width"] * (slider["max"] - slider["min"]) + slider["min"]
                    # Clamp value to slider range
                    slider["value"] = max(slider["min"], min(slider["max"], new_val))
    
    # Draw current state
    if state == MENU:
        draw_menu()
    elif state == TUTORIAL:
        draw_tutorial()
        # Draw back button in tutorial mode
        back_button = pygame.Rect(30, 10, 100, 40)
        pygame.draw.rect(screen, LIGHT_RED, back_button, border_radius=5)
        draw_text("Menu", font_small, BLACK, back_button.centerx, back_button.centery, "center")
    elif state == GAME:
        draw_game(layout, mouse_pos)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()