import pygame
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Required for pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from scipy import stats
import sys
import os

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
LIGHT_BLUE = (173, 216, 230)
LIGHT_GREEN = (144, 238, 144)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)

# Set up the display
WIDTH, HEIGHT = 1200, 800
MIN_WIDTH, MIN_HEIGHT = 1000, 700  # Minimum window dimensions to prevent layout issues
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("One-Sample T-Test Visualization for Alzheimer's Research")

# Set window icon if available
try:
    icon = pygame.image.load("brain_icon.png")
    pygame.display.set_icon(icon)
except:
    pass  # If icon file is not available, skip it

# Fonts
font_small = pygame.font.SysFont('Arial', 16)
font_medium = pygame.font.SysFont('Arial', 20)
font_large = pygame.font.SysFont('Arial', 24)
font_title = pygame.font.SysFont('Arial', 32, bold=True)

# Data for one-sample t-test (example data provided)
original_data = np.array([86, 73, 50, 73, 24, 65, 84, 54, 16, 26])  # Original data
data = original_data.copy()  # Working copy
hypothesized_mean = 43.4  # Default test value (mu)

# For interactive adjustments
selected_point = None
dragging = False
show_info = True
significance_level = 0.05  # Default alpha level

# Define initial buttons and UI elements - these will be updated in calculate_layout()
reset_button = pygame.Rect(0, 0, 120, 40)
info_button = pygame.Rect(0, 0, 120, 40)
test_value_slider = {"rect": pygame.Rect(0, 0, 300, 10), "pos": 0, "dragging": False}

# Layout elements to be calculated
layout = {}

def calculate_layout():
    """
    Calculate and update all UI component positions based on current window size
    to ensure proper spacing and prevent overlaps.
    """
    global reset_button, info_button, test_value_slider, layout
    
    # Calculate header section
    header_height = 60
    
    # Calculate main sections with proper spacing
    sidebar_width = max(350, WIDTH * 0.3)
    content_width = WIDTH - (2 * sidebar_width) - 40  # 40px for margins
    
    # Ensure minimum dimensions for visualization area
    if content_width < 400:
        content_width = 400
        sidebar_width = (WIDTH - content_width - 40) / 2
    
    # Calculate visualization section (middle column)
    vis_left = sidebar_width + 20
    vis_width = content_width
    vis_top = header_height + 20
    vis_height = HEIGHT - vis_top - 120  # 120px for bottom controls
    
    # Split visualization vertically for plot and test value slider
    vis_plot_height = vis_height - 60
    
    # Calculate left panel (t-test info)
    info_panel_left = 20
    info_panel_top = header_height + 20
    info_panel_width = sidebar_width - 20
    info_panel_height = vis_height // 2
    
    # Calculate right panel (data points)
    data_panel_left = vis_left + vis_width + 20
    data_panel_top = header_height + 20
    data_panel_width = sidebar_width - 40
    data_panel_height = vis_height
    
    # Calculate bottom controls
    controls_top = HEIGHT - 100
    
    # Update buttons and slider positions
    info_button = pygame.Rect(WIDTH - 150, 20, 120, 40)
    reset_button = pygame.Rect(WIDTH - 150, controls_top, 120, 40)
    
    # Test value slider - position at bottom of visualization area
    slider_left = vis_left + 50
    slider_width = vis_width - 100
    slider_top = controls_top + 15
    
    test_value_slider = {
        "rect": pygame.Rect(slider_left, slider_top, slider_width, 10),
        "pos": slider_left + (slider_width * (hypothesized_mean - 0) / 100),
        "dragging": test_value_slider["dragging"]
    }
    
    # Update layout dictionary
    layout = {
        "header": {
            "top": 0,
            "height": header_height
        },
        "info_panel": {
            "left": info_panel_left,
            "top": info_panel_top,
            "width": info_panel_width,
            "height": info_panel_height
        },
        "data_panel": {
            "left": data_panel_left,
            "top": data_panel_top,
            "width": data_panel_width,
            "height": data_panel_height
        },
        "visualization": {
            "left": vis_left,
            "top": vis_top,
            "width": vis_width,
            "height": vis_plot_height
        },
        "controls": {
            "top": controls_top
        }
    }
    
    return layout

def perform_ttest(data, mu):
    """Calculate t-test statistics and p-values"""
    # Basic statistics
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # Using n-1 for sample std dev
    sample_size = len(data)
    standard_error = sample_std / np.sqrt(sample_size)
    
    # Perform one-sample t-test
    t_stat, p_two_sided = stats.ttest_1samp(data, mu)
    
    # For one-sided tests, we need to adjust the p-value
    p_greater = p_two_sided / 2 if t_stat > 0 else 1 - (p_two_sided / 2)
    p_less = p_two_sided / 2 if t_stat < 0 else 1 - (p_two_sided / 2)
    
    # Calculate confidence interval
    ci_lower = sample_mean - stats.t.ppf(0.975, df=sample_size-1) * standard_error
    ci_upper = sample_mean + stats.t.ppf(0.975, df=sample_size-1) * standard_error
    
    return {
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "t_stat": t_stat,
        "p_two_sided": p_two_sided,
        "p_greater": p_greater,
        "p_less": p_less,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant_two_sided": p_two_sided < significance_level,
        "significant_greater": p_greater < significance_level,
        "significant_less": p_less < significance_level
    }

def create_visualization_surface(data, mu, test_results, width=800, height=500):
    """Create a matplotlib visualization as a pygame surface"""
    # Create figure with proper dimensions to fit in layout
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    
    # Set up a 1x2 grid of subplots with appropriate spacing
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.3)
    
    # Plot 1: Raw data with boxplot
    ax1 = fig.add_subplot(121)
    ax1.boxplot(data, widths=0.6)
    
    # Add scatter plot of individual points - jitter points horizontally for better visibility
    x_jitter = np.random.normal(1, 0.04, size=len(data))
    ax1.scatter(x_jitter, data, color='blue', alpha=0.6)
    
    # Draw test value and sample mean lines
    ax1.axhline(y=mu, color='r', linestyle='--', label=f'Test Value (μ={mu:.1f})')
    ax1.axhline(y=test_results["sample_mean"], color='g', linestyle='-', 
                label=f'Sample Mean ({test_results["sample_mean"]:.1f})')
    
    # Add confidence interval
    ax1.axhspan(test_results["ci_lower"], test_results["ci_upper"], alpha=0.2, color='green', 
                label=f'95% CI [{test_results["ci_lower"]:.1f}, {test_results["ci_upper"]:.1f}]')
    
    ax1.set_ylabel('Value')
    ax1.set_title('Data Comparison to Test Value')
    ax1.legend(loc='upper right', fontsize=8)
    
    # Plot 2: T-distribution with critical regions
    ax2 = fig.add_subplot(122)
    
    # Create t-distribution
    df = len(data) - 1
    x = np.linspace(-5, 5, 1000)
    y = stats.t.pdf(x, df)
    
    # Plot the curve
    ax2.plot(x, y, 'b-', lw=2, label=f't-distribution (df={df})')
    
    # Add the critical regions for two-sided test
    critical_t = stats.t.ppf(1 - significance_level/2, df)
    ax2.fill_between(x, 0, y, where=(x >= critical_t) | (x <= -critical_t), 
                    color='pink', alpha=0.5, label=f'Critical Regions (α={significance_level})')
    
    # Add observed t-statistic
    ax2.axvline(x=test_results["t_stat"], color='r', linestyle='-', lw=1.5, 
                label=f't-stat = {test_results["t_stat"]:.3f}\np = {test_results["p_two_sided"]:.4f}')
    
    ax2.set_xlabel('t-value')
    ax2.set_ylabel('Density')
    ax2.set_title('T-Distribution with Observed Statistic')
    ax2.legend(loc='upper right', fontsize=8)
    
    # Convert matplotlib figure to pygame surface
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    
    # Use buffer_rgba instead of tostring_rgb (which is deprecated)
    raw_data = renderer.buffer_rgba()
    size = canvas.get_width_height()
    
    plt.close(fig)
    
    # Create surface from buffer
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    return surf

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

def draw_slider(slider, value, min_val, max_val, label):
    """Draw a slider with current value"""
    # Draw slider track
    pygame.draw.rect(screen, GRAY, slider["rect"])
    
    # Calculate handle position
    handle_pos = slider["rect"].left + ((value - min_val) / (max_val - min_val)) * slider["rect"].width
    
    # Draw handle
    handle_radius = 12
    pygame.draw.circle(screen, BLUE, (int(handle_pos), slider["rect"].centery), handle_radius)
    
    # Draw label
    draw_text(f"{label}: {value:.1f}", font_medium, BLACK, 
              slider["rect"].left, slider["rect"].top - 15, "left")
    
    # Draw min and max values
    draw_text(f"{min_val}", font_small, DARK_GRAY, 
              slider["rect"].left, slider["rect"].bottom + 15, "left")
    draw_text(f"{max_val}", font_small, DARK_GRAY, 
              slider["rect"].right, slider["rect"].bottom + 15, "right")

def draw_info_panel(test_results):
    """Draw information panel with t-test results"""
    if show_info:
        info_panel = pygame.Rect(
            layout["info_panel"]["left"],
            layout["info_panel"]["top"],
            layout["info_panel"]["width"],
            layout["info_panel"]["height"]
        )
        pygame.draw.rect(screen, LIGHT_BLUE, info_panel, border_radius=10)
        pygame.draw.rect(screen, BLUE, info_panel, width=2, border_radius=10)
        
        # Draw title
        draw_text("T-Test Analysis", font_large, BLACK, info_panel.left + 10, info_panel.top + 20)
        
        # Draw statistics
        stats_y = info_panel.top + 60
        line_height = 25
        
        draw_text(f"Sample Size: {len(data)}", font_medium, BLACK, 
                  info_panel.left + 15, stats_y)
        stats_y += line_height
        
        draw_text(f"Sample Mean: {test_results['sample_mean']:.2f}", font_medium, BLACK, 
                  info_panel.left + 15, stats_y)
        stats_y += line_height
        
        draw_text(f"Sample Std Dev: {test_results['sample_std']:.2f}", font_medium, BLACK, 
                  info_panel.left + 15, stats_y)
        stats_y += line_height
        
        draw_text(f"Standard Error: {test_results['standard_error']:.2f}", font_medium, BLACK, 
                  info_panel.left + 15, stats_y)
        stats_y += line_height
        
        # Draw line separator
        pygame.draw.line(screen, BLUE, 
                         (info_panel.left + 15, stats_y), 
                         (info_panel.right - 15, stats_y), 2)
        stats_y += 10
        
        # Draw test results with color coding for significance
        t_color = RED if test_results["significant_two_sided"] else BLACK
        draw_text(f"T-statistic: {test_results['t_stat']:.3f}", font_medium, BLACK, 
                  info_panel.left + 15, stats_y)
        stats_y += line_height
        
        draw_text(f"P-value (two-sided): {test_results['p_two_sided']:.4f}", 
                  font_medium, t_color, info_panel.left + 15, stats_y)
        stats_y += line_height
        
        draw_text(f"Significant (α={significance_level}):", font_medium, BLACK, 
                  info_panel.left + 15, stats_y)
        
        # Add colored indicator for significance
        indicator_rect = pygame.Rect(info_panel.left + 170, stats_y - 10, 20, 20)
        indicator_color = GREEN if test_results["significant_two_sided"] else RED
        pygame.draw.rect(screen, indicator_color, indicator_rect)
        
        # Draw interpretation
        stats_y += line_height + 5
        if test_results["significant_two_sided"]:
            draw_text("Result: Mean differs from test value", 
                     font_medium, GREEN, info_panel.left + 15, stats_y)
        else:
            draw_text("Result: No significant difference", 
                     font_medium, RED, info_panel.left + 15, stats_y)

def draw_data_table():
    """Draw a table showing the current data values with improved layout"""
    table_rect = pygame.Rect(
        layout["data_panel"]["left"],
        layout["data_panel"]["top"],
        layout["data_panel"]["width"],
        layout["data_panel"]["height"]
    )
    pygame.draw.rect(screen, LIGHT_GREEN, table_rect, border_radius=10)
    pygame.draw.rect(screen, GREEN, table_rect, width=2, border_radius=10)
    
    # Draw title
    draw_text("Data Points (Drag to Modify)", font_large, BLACK, table_rect.left + 10, table_rect.top + 20)
    
    # Calculate layout for data display
    # Determine optimal columns based on table width
    available_width = table_rect.width - 20
    min_cell_width = 80
    cols = max(1, min(5, available_width // min_cell_width))
    
    rows = int(np.ceil(len(data) / cols))
    cell_width = available_width // cols
    cell_height = 30
    start_y = table_rect.top + 60
    
    # Ensure spacing between rows is reasonable
    max_visible_rows = (table_rect.height - 80) // cell_height
    if rows > max_visible_rows:
        cell_height = (table_rect.height - 80) // rows
    
    for i, value in enumerate(data):
        col = i % cols
        row = i // cols
        
        cell_x = table_rect.left + 10 + col * cell_width
        cell_y = start_y + row * cell_height
        
        # Draw highlighting if this is the currently selected point
        if selected_point == i:
            highlight_rect = pygame.Rect(cell_x - 5, cell_y - 15, cell_width, cell_height)
            pygame.draw.rect(screen, GREEN, highlight_rect, border_radius=5)
        
        # Display data value
        draw_text(f"{i+1}: {value:.1f}", font_medium, BLACK, cell_x, cell_y)
        
        # Draw interactive point - to the right of the text
        point_rect = pygame.Rect(cell_x + 60, cell_y - 10, 20, 20)
        pygame.draw.rect(screen, BLUE, point_rect, border_radius=5)

def main():
    global selected_point, dragging, show_info, hypothesized_mean, data
    global test_value_slider, reset_button, info_button, WIDTH, HEIGHT, screen, layout
    
    # Initialize layout
    calculate_layout()
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.VIDEORESIZE:
                # Update window size and reposition elements
                # Enforce minimum dimensions
                WIDTH = max(event.w, MIN_WIDTH)
                HEIGHT = max(event.h, MIN_HEIGHT)
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
                
                # Recalculate layout
                calculate_layout()
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                
                # Check if clicked on the reset button
                if reset_button.collidepoint(mouse_pos):
                    data = original_data.copy()
                    selected_point = None
                
                # Check if clicked on the info toggle button
                elif info_button.collidepoint(mouse_pos):
                    show_info = not show_info
                
                # Check for slider interaction
                slider_rect_expanded = pygame.Rect(
                    test_value_slider["rect"].left - 10,
                    test_value_slider["rect"].top - 10,
                    test_value_slider["rect"].width + 20,
                    test_value_slider["rect"].height + 20
                )
                if slider_rect_expanded.collidepoint(mouse_pos):
                    test_value_slider["dragging"] = True
                    # Update slider position and value immediately
                    test_value_slider["pos"] = mouse_pos[0]
                    hypothesized_mean = max(0, min(100, (test_value_slider["pos"] - test_value_slider["rect"].left) / 
                                               test_value_slider["rect"].width * 100))
                
                # Check if clicked on a data point in the table
                data_panel = pygame.Rect(
                    layout["data_panel"]["left"],
                    layout["data_panel"]["top"],
                    layout["data_panel"]["width"],
                    layout["data_panel"]["height"]
                )
                
                if data_panel.collidepoint(mouse_pos):
                    # Calculate which data point was clicked - using improved layout calculation
                    relative_y = mouse_pos[1] - (data_panel.top + 60)
                    relative_x = mouse_pos[0] - (data_panel.left + 10)
                    
                    available_width = data_panel.width - 20
                    min_cell_width = 80
                    cols = max(1, min(5, available_width // min_cell_width))
                    cell_width = available_width // cols
                    cell_height = 30
                    
                    # Adjust cell_height for smaller panels
                    max_visible_rows = (data_panel.height - 80) // cell_height
                    rows = int(np.ceil(len(data) / cols))
                    if rows > max_visible_rows:
                        cell_height = (data_panel.height - 80) // rows
                    
                    col = relative_x // cell_width
                    row = relative_y // cell_height
                    
                    point_index = row * cols + col
                    if 0 <= point_index < len(data):
                        selected_point = point_index
                        dragging = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
                test_value_slider["dragging"] = False
            
            elif event.type == pygame.MOUSEMOTION:
                if dragging and selected_point is not None:
                    # Update the selected data point value based on vertical mouse movement
                    # Invert the movement (up = increase, down = decrease)
                    data[selected_point] = max(0, data[selected_point] - event.rel[1])
                
                if test_value_slider["dragging"]:
                    # Update slider position and hypothesized mean
                    test_value_slider["pos"] = max(test_value_slider["rect"].left, 
                                               min(test_value_slider["rect"].right, pygame.mouse.get_pos()[0]))
                    hypothesized_mean = max(0, min(100, (test_value_slider["pos"] - test_value_slider["rect"].left) / 
                                               test_value_slider["rect"].width * 100))
        
        # Calculate t-test results
        test_results = perform_ttest(data, hypothesized_mean)
        
        # Clear the screen
        screen.fill(WHITE)
        
        # Draw title in header
        draw_text("One-Sample T-Test Visualization for Alzheimer's Research", 
                 font_title, BLUE, WIDTH // 2, 30, "center")
        
        # Draw visualization
        try:
            vis_width = layout["visualization"]["width"] - 20  # Padding
            vis_height = layout["visualization"]["height"] - 20  # Padding
            
            vis_surface = create_visualization_surface(data, hypothesized_mean, test_results, 
                                                      width=vis_width, height=vis_height)
            
            vis_rect = vis_surface.get_rect(
                center=(layout["visualization"]["left"] + layout["visualization"]["width"]//2,
                       layout["visualization"]["top"] + layout["visualization"]["height"]//2)
            )
            
            screen.blit(vis_surface, vis_rect)
        except Exception as e:
            # Handle potential visualization errors
            error_text = f"Error creating visualization: {str(e)}"
            draw_text(error_text, font_medium, RED, WIDTH//2, HEIGHT//2, "center")
            print(f"Visualization error: {e}")
        
        # Draw info panel
        draw_info_panel(test_results)
        
        # Draw data table
        draw_data_table()
        
        # Draw reset button
        pygame.draw.rect(screen, LIGHT_BLUE, reset_button, border_radius=5)
        draw_text("Reset Data", font_medium, BLACK, reset_button.centerx, reset_button.centery, "center")
        
        # Draw info toggle button
        info_button_color = LIGHT_GREEN if show_info else LIGHT_BLUE
        pygame.draw.rect(screen, info_button_color, info_button, border_radius=5)
        draw_text("Toggle Info", font_medium, BLACK, info_button.centerx, info_button.centery, "center")
        
        # Draw slider for test value
        draw_slider(test_value_slider, hypothesized_mean, 0, 100, "Test Value (μ)")
        
        # Instructions
        instructions = [
            "Instructions:",
            "• Click and drag data points in the table to modify values",
            "• Adjust the 'Test Value' slider to change the hypothesized mean",
            "• Click 'Reset Data' to restore original values",
            "• Click 'Toggle Info' to show/hide analysis details"
        ]
        
        for i, line in enumerate(instructions):
            draw_text(line, font_small, DARK_GRAY, 20, HEIGHT - 120 + i*20)
        
        # Update the display
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()