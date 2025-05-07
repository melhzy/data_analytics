import pygame
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Required for pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from scipy import stats
import sys

# Initialize pygame
pygame.init()

# Colors - Normalized for matplotlib (0-1 range)
MPL_LIGHT_BLUE = (173/255, 216/255, 230/255)
MPL_LIGHT_RED = (255/255, 182/255, 193/255)

# Colors for PyGame (0-255 range)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 180, 0)
DARK_GREEN = (0, 100, 0)
LIGHT_BLUE = (173, 216, 230)
LIGHT_GREEN = (144, 238, 144)
LIGHT_RED = (255, 182, 193)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
DARK_RED = (180, 0, 0)

# Set up the display
WIDTH, HEIGHT = 1200, 800
MIN_WIDTH, MIN_HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Paired Samples T-Test for Alzheimer's Research")

# Try to load an icon if available
try:
    icon = pygame.image.load("brain_icon.png")
    pygame.display.set_icon(icon)
except:
    pass  # Skip if icon file is not available

# Fonts
font_small = pygame.font.SysFont('Arial', 16)
font_medium = pygame.font.SysFont('Arial', 20)
font_large = pygame.font.SysFont('Arial', 24)
font_title = pygame.font.SysFont('Arial', 32, bold=True)

# Sample data for Alzheimer's biomarker measurements (e.g., Amyloid-β levels in CSF)
original_before = np.array([42.5, 38.7, 51.2, 45.8, 49.3, 55.1, 41.2, 47.6, 52.3, 44.9])
original_after = np.array([35.2, 32.1, 45.7, 42.3, 41.8, 49.5, 38.6, 40.2, 48.1, 40.5])

before_data = original_before.copy()
after_data = original_after.copy()

# Create subject IDs array
subject_ids = np.arange(1, len(before_data) + 1)

# For interactive adjustments
selected_point = None
selected_column = None  # "before" or "after"
dragging = False
show_info = True
significance_level = 0.05  # Default alpha level

# Define initial buttons and UI elements
reset_button = pygame.Rect(0, 0, 120, 40)
info_button = pygame.Rect(0, 0, 120, 40)

# Layout elements to be calculated
layout = {}

def calculate_layout():
    """Calculate UI component positions based on current window size"""
    global reset_button, info_button, layout
    
    # Calculate header section
    header_height = 60
    
    # Calculate main sections with proper spacing
    sidebar_width = max(350, WIDTH * 0.3)
    content_width = WIDTH - (2 * sidebar_width) - 40
    
    # Ensure minimum dimensions for visualization area
    if content_width < 400:
        content_width = 400
        sidebar_width = (WIDTH - content_width - 40) / 2
    
    # Calculate visualization section (middle column)
    vis_left = sidebar_width + 20
    vis_width = content_width
    vis_top = header_height + 20
    vis_height = HEIGHT - vis_top - 120
    
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
    
    # Update buttons positions
    info_button = pygame.Rect(WIDTH - 150, 20, 120, 40)
    reset_button = pygame.Rect(WIDTH - 150, controls_top, 120, 40)
    
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
            "height": vis_height
        },
        "controls": {
            "top": controls_top
        }
    }
    
    return layout

def perform_paired_ttest(before, after):
    """Calculate paired t-test statistics and p-values for Alzheimer's biomarker data"""
    # Calculate differences
    differences = before - after
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(before, after, alternative='two-sided')
    
    # Calculate basic statistics
    mean_before = np.mean(before)
    mean_after = np.mean(after)
    std_before = np.std(before, ddof=1)
    std_after = np.std(after, ddof=1)
    
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(len(differences))
    
    # Calculate confidence interval
    df = len(differences) - 1
    ci_95 = stats.t.ppf([0.025, 0.975], df) * se_diff + mean_diff
    
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "std_before": std_before,
        "std_after": std_after,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "ci_95": ci_95,
        "differences": differences,
        "significant": p_value < significance_level
    }

def create_visualization_surface(before, after, test_results, width=800, height=500):
    """Create a matplotlib visualization of Alzheimer's biomarker data as a pygame surface"""
    # Create figure with proper dimensions to fit in layout
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    
    # Set up a 2x2 grid of subplots with appropriate spacing
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.3, hspace=0.4)
    
    # 1. Boxplot comparison
    ax1 = fig.add_subplot(221)
    ax1.set_title('Biomarker Distribution Comparison')
    boxplot_data = [before, after]
    box = ax1.boxplot(boxplot_data, patch_artist=True)
    
    # Set colors for boxplots with normalized values (0-1)
    box['boxes'][0].set_facecolor(MPL_LIGHT_BLUE)
    box['boxes'][1].set_facecolor(MPL_LIGHT_RED)
    
    ax1.set_xticklabels(['Pre-Treatment', 'Post-Treatment'])
    ax1.set_ylabel('Amyloid-β Level (pg/mL)')
    
    # Add means as dashed lines
    ax1.axhline(y=test_results["mean_before"], color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=test_results["mean_after"], color='red', linestyle='--', alpha=0.5)
    
    # 2. Individual subject changes
    ax2 = fig.add_subplot(222)
    ax2.set_title('Individual Patient Changes')
    for i in range(len(before)):
        ax2.plot([1, 2], [before[i], after[i]], 'o-', alpha=0.5)
    ax2.set_xlim(0.5, 2.5)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Pre-Treatment', 'Post-Treatment'])
    ax2.set_ylabel('Amyloid-β Level (pg/mL)')
    
    # Add mean lines
    ax2.plot([1, 2], [test_results["mean_before"], test_results["mean_after"]], 
             'r--', linewidth=2, label='Mean')
    ax2.legend()
    
    # 3. Difference histogram
    ax3 = fig.add_subplot(223)
    ax3.set_title('Differences (Pre - Post)')
    ax3.hist(test_results["differences"], bins=7, alpha=0.7, color=(0.5, 0.7, 0.9))
    
    # Add vertical lines for mean difference and CI
    ax3.axvline(test_results["mean_diff"], color='red', linestyle='--', 
                label=f'Mean: {test_results["mean_diff"]:.2f}')
    ax3.axvline(test_results["ci_95"][0], color='green', linestyle=':', 
                label=f'95% CI: [{test_results["ci_95"][0]:.2f}, {test_results["ci_95"][1]:.2f}]')
    ax3.axvline(test_results["ci_95"][1], color='green', linestyle=':')
    ax3.axvline(0, color='black', linestyle='-', label='No Effect')
    ax3.legend(fontsize=8)
    ax3.set_xlabel('Difference (Pre - Post)')
    
    # 4. Scatter plot of Before vs After
    ax4 = fig.add_subplot(224)
    ax4.set_title('Pre vs. Post Comparison')
    ax4.scatter(before, after)
    
    # Add identity line (y=x)
    min_val = min(np.min(before), np.min(after))
    max_val = max(np.max(before), np.max(after))
    padding = (max_val - min_val) * 0.1
    ax4.plot([min_val-padding, max_val+padding], [min_val-padding, max_val+padding], 
             'k--', alpha=0.5, label='No Change Line')
    
    # Points above line = increase, below = decrease
    ax4.text(min_val, max_val, "Increased", fontsize=8, ha='left', va='top')
    ax4.text(max_val, min_val, "Decreased", fontsize=8, ha='right', va='bottom')
    
    ax4.set_xlabel('Pre-Treatment Level')
    ax4.set_ylabel('Post-Treatment Level')
    ax4.legend(fontsize=8)
    
    # Add overall title with alzheimer's focus
    fig.suptitle('Alzheimer\'s Biomarker Analysis', fontsize=14)
    
    # Convert matplotlib figure to pygame surface
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    
    # Use buffer_rgba instead of tostring_rgb
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

def draw_info_panel(test_results):
    """Draw information panel with paired t-test results for Alzheimer's study"""
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
        draw_text("Paired T-Test Results", font_large, BLACK, info_panel.left + 10, info_panel.top + 20)
        
        # Draw statistics
        stats_y = info_panel.top + 60
        line_height = 25
        
        draw_text(f"Sample Size: {len(before_data)}", font_medium, BLACK, 
                  info_panel.left + 15, stats_y)
        stats_y += line_height
        
        draw_text(f"Mean Pre-Treatment: {test_results['mean_before']:.2f}", font_medium, BLACK, 
                  info_panel.left + 15, stats_y)
        stats_y += line_height
        
        draw_text(f"Mean Post-Treatment: {test_results['mean_after']:.2f}", font_medium, BLACK, 
                  info_panel.left + 15, stats_y)
        stats_y += line_height
        
        draw_text(f"Mean Difference: {test_results['mean_diff']:.2f}", font_medium, BLACK, 
                  info_panel.left + 15, stats_y)
        stats_y += line_height
        
        # Draw line separator
        pygame.draw.line(screen, BLUE, 
                         (info_panel.left + 15, stats_y), 
                         (info_panel.right - 15, stats_y), 2)
        stats_y += 10
        
        # Draw test results with color coding for significance
        t_color = DARK_RED if test_results["significant"] else BLACK
        draw_text(f"T-statistic: {test_results['t_stat']:.3f}", font_medium, BLACK, 
                  info_panel.left + 15, stats_y)
        stats_y += line_height
        
        draw_text(f"P-value: {test_results['p_value']:.4f}", 
                  font_medium, t_color, info_panel.left + 15, stats_y)
        stats_y += line_height
        
        draw_text(f"95% CI: [{test_results['ci_95'][0]:.2f}, {test_results['ci_95'][1]:.2f}]",
                 font_medium, BLACK, info_panel.left + 15, stats_y)
        stats_y += line_height
        
        draw_text(f"Significant (α={significance_level}):", font_medium, BLACK, 
                  info_panel.left + 15, stats_y)
        
        # Add colored indicator for significance
        indicator_rect = pygame.Rect(info_panel.left + 170, stats_y - 10, 20, 20)
        indicator_color = DARK_GREEN if test_results["significant"] else DARK_RED
        pygame.draw.rect(screen, indicator_color, indicator_rect)
        
        # Draw interpretation
        stats_y += line_height + 5
        if test_results["significant"]:
            direction = "decrease" if test_results["mean_diff"] > 0 else "increase"
            draw_text(f"Result: Significant {direction} in Amyloid-β", 
                     font_medium, DARK_GREEN, info_panel.left + 15, stats_y)
        else:
            draw_text("Result: No significant difference", 
                     font_medium, DARK_RED, info_panel.left + 15, stats_y)

def draw_data_table():
    """Draw a table showing the paired Alzheimer's biomarker data"""
    table_rect = pygame.Rect(
        layout["data_panel"]["left"],
        layout["data_panel"]["top"],
        layout["data_panel"]["width"],
        layout["data_panel"]["height"]
    )
    pygame.draw.rect(screen, LIGHT_GREEN, table_rect, border_radius=10)
    pygame.draw.rect(screen, GREEN, table_rect, width=2, border_radius=10)
    
    # Draw title
    draw_text("Amyloid-β Measurements (pg/mL)", font_large, BLACK, 
              table_rect.left + 10, table_rect.top + 20)
    
    # Calculate layout for data display
    start_y = table_rect.top + 60
    title_y = start_y + 10
    
    # Draw column headers
    before_header_x = table_rect.left + table_rect.width * 0.25
    after_header_x = table_rect.left + table_rect.width * 0.75
    
    draw_text("Pre-Treatment", font_medium, BLUE, before_header_x, title_y, "center")
    draw_text("Post-Treatment", font_medium, RED, after_header_x, title_y, "center")
    
    # Draw separator line below headers
    pygame.draw.line(screen, BLACK, 
                    (table_rect.left + 10, title_y + 20), 
                    (table_rect.right - 10, title_y + 20), 1)
    
    # Data rows
    data_start_y = title_y + 35
    row_height = min(30, (table_rect.height - 100) / len(before_data))
    
    # Display all subject data
    for i in range(len(before_data)):
        # Get subject number from the dedicated array
        subject_number = subject_ids[i]
        
        # Row y-position
        row_y = data_start_y + i * row_height
        
        # Draw subject label with explicit number
        subject_label = f"Patient {subject_number}:"
        draw_text(subject_label, font_medium, BLACK, table_rect.left + 10, row_y)
        
        # Add alternating row background for better readability
        if i % 2 == 1:
            row_bg = pygame.Rect(table_rect.left + 10, row_y - 15, table_rect.width - 20, row_height)
            pygame.draw.rect(screen, (200, 230, 200), row_bg)  # Light green background
        
        # Highlight selected row
        if selected_point == i:
            highlight_rect = pygame.Rect(
                table_rect.left + 10, row_y - 15, 
                table_rect.width - 20, row_height
            )
            pygame.draw.rect(screen, LIGHT_BLUE if selected_column == "before" else LIGHT_RED, 
                            highlight_rect, border_radius=3)
        
        # Before value cell with improved contrast
        before_diff_color = DARK_GREEN if before_data[i] > after_data[i] else DARK_RED
        cell_before = pygame.Rect(
            before_header_x - 25, row_y - 12, 
            50, 24
        )
        pygame.draw.rect(screen, WHITE, cell_before, border_radius=3)
        pygame.draw.rect(screen, BLUE, cell_before, width=2, border_radius=3)
        draw_text(f"{before_data[i]:.1f}", font_medium, BLACK, before_header_x, row_y, "center")
        
        # After value cell with improved contrast
        after_diff_color = DARK_RED if before_data[i] > after_data[i] else DARK_GREEN
        cell_after = pygame.Rect(
            after_header_x - 25, row_y - 12, 
            50, 24
        )
        pygame.draw.rect(screen, WHITE, cell_after, border_radius=3)
        pygame.draw.rect(screen, RED, cell_after, width=2, border_radius=3)
        draw_text(f"{after_data[i]:.1f}", font_medium, BLACK, after_header_x, row_y, "center")
        
        # Draw difference with improved contrast
        diff = before_data[i] - after_data[i]
        diff_x = (after_header_x + table_rect.right) / 2
        diff_color = DARK_GREEN if diff > 0 else DARK_RED
        diff_text = f"{diff:.1f}"
        draw_text(diff_text, font_medium, diff_color, diff_x, row_y, "center")

def main():
    global selected_point, selected_column, dragging, show_info, before_data, after_data
    global reset_button, info_button, layout, screen, WIDTH, HEIGHT
    
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
                    before_data = original_before.copy()
                    after_data = original_after.copy()
                    selected_point = None
                    selected_column = None
                
                # Check if clicked on the info toggle button
                elif info_button.collidepoint(mouse_pos):
                    show_info = not show_info
                
                # Check if clicked on a data cell in the table
                data_panel = pygame.Rect(
                    layout["data_panel"]["left"],
                    layout["data_panel"]["top"],
                    layout["data_panel"]["width"],
                    layout["data_panel"]["height"]
                )
                
                if data_panel.collidepoint(mouse_pos):
                    # Calculate coordinates within the table
                    title_y = data_panel.top + 60 + 10
                    data_start_y = title_y + 35
                    row_height = min(30, (data_panel.height - 100) / len(before_data))
                    
                    # Check if in data area
                    if mouse_pos[1] >= data_start_y:
                        # Calculate row
                        row_index = int((mouse_pos[1] - data_start_y) / row_height)
                        if 0 <= row_index < len(before_data):
                            selected_point = row_index
                            
                            # Check if in "before" or "after" column
                            before_header_x = data_panel.left + data_panel.width * 0.25
                            after_header_x = data_panel.left + data_panel.width * 0.75
                            
                            # Determine column based on x position
                            if abs(mouse_pos[0] - before_header_x) < abs(mouse_pos[0] - after_header_x):
                                selected_column = "before"
                            else:
                                selected_column = "after"
                            
                            dragging = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if dragging and selected_point is not None and selected_column is not None:
                    # Update the selected data point value based on vertical mouse movement
                    # Invert the movement (up = increase, down = decrease)
                    change = -event.rel[1] * 0.5  # Scale the movement
                    
                    if selected_column == "before":
                        before_data[selected_point] = max(0, before_data[selected_point] + change)
                    else:  # "after"
                        after_data[selected_point] = max(0, after_data[selected_point] + change)
        
        # Calculate t-test results
        test_results = perform_paired_ttest(before_data, after_data)
        
        # Clear the screen
        screen.fill(WHITE)
        
        # Draw title in header
        draw_text("Alzheimer's Biomarker Analysis: Amyloid-β Levels Pre & Post Treatment", 
                 font_title, BLUE, WIDTH // 2, 30, "center")
        
        # Draw visualization
        try:
            vis_width = layout["visualization"]["width"] - 20
            vis_height = layout["visualization"]["height"] - 20
            
            vis_surface = create_visualization_surface(before_data, after_data, test_results, 
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
        
        # Instructions
        instructions = [
            "Instructions:",
            "• Click and drag cells in the table to modify biomarker values",
            "• Green difference values indicate improvement (decrease in Amyloid-β)",
            "• Red difference values indicate worsening (increase in Amyloid-β)",
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