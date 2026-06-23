import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import ttk

# =====================================================================
# STEP 1: GENERATE AND FORMAT THE DATASET
# =====================================================================
X_raw, y_raw = make_regression(n_samples=1000, n_features=3, noise=15.0, random_state=42)

df = pd.DataFrame(X_raw, columns=['Size_SqFt', 'Bedrooms', 'Neighborhood_Score'])
df['Size_SqFt'] = (df['Size_SqFt'] * 500) + 2000          # Ranges roughly from 500 to 3500 SqFt
df['Bedrooms'] = np.clip(((df['Bedrooms'] * 1) + 3).astype(int), 1, 5) # Limits bedrooms between 1 and 5
df['Neighborhood_Score'] = (df['Neighborhood_Score'] * 2) + 5 # Scale score to a 1-10 range
df['Price'] = (df['Size_SqFt'] * 150) + (df['Bedrooms'] * 25000) + (df['Neighborhood_Score'] * 15000) + (y_raw * 100)

# =====================================================================
# STEP 2: SPLIT DATA INTO TRAINING & TESTING SETS
# =====================================================================
X = df[['Size_SqFt', 'Bedrooms', 'Neighborhood_Score']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================================================
# STEP 3: FEATURE SCALING
# =====================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================================
# STEP 4: INSTANTIATE AND TRAIN THE MODEL
# =====================================================================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# =====================================================================
# STEP 5: EVALUATE PERFORMANCE
# =====================================================================
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Calculate unscaled coefficients and intercept for formula displaying
# Price = Size_SqFt * w1_unscaled + Bedrooms * w2_unscaled + Neighborhood_Score * w3_unscaled + intercept_unscaled
unscaled_coefs = model.coef_ / scaler.scale_
intercept = model.intercept_ - np.sum((model.coef_ * scaler.mean_) / scaler.scale_)

# =====================================================================
# STEP 6: BUILD THE TKINTER DASHBOARD
# =====================================================================
class HousePredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Premium House Price Predictor")
        self.root.configure(bg="#121212")
        
        # Centered window layout size
        width, height = 1250, 750
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(1050, 650)
        
        # Responsive grid weights
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        # Main container with border padding
        self.main_container = tk.Frame(self.root, bg="#121212", padx=20, pady=20)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        self.main_container.columnconfigure(0, weight=3) # Controls (30%)
        self.main_container.columnconfigure(1, weight=7) # Visualizations (70%)
        self.main_container.rowconfigure(0, weight=1)
        
        # Initialize sub-panels
        self.setup_left_panel()
        self.setup_right_panel()
        
        # Draw initial state prediction
        self.update_prediction()

    def setup_left_panel(self):
        # Left pane frame
        self.left_pane = tk.Frame(
            self.main_container, bg="#1e1e1e", padx=25, pady=25,
            highlightthickness=1, highlightbackground="#2d2d2d", bd=0
        )
        self.left_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        
        # Title and description
        title_label = tk.Label(
            self.left_pane, text="🏠 House Evaluator", 
            font=("Segoe UI", 20, "bold"), fg="#ffffff", bg="#1e1e1e"
        )
        title_label.pack(anchor="w", pady=(0, 2))
        
        desc_label = tk.Label(
            self.left_pane, text="A Machine Learning House Value Predictor", 
            font=("Segoe UI", 10), fg="#888888", bg="#1e1e1e"
        )
        desc_label.pack(anchor="w", pady=(0, 25))
        
        # 1. Size Slider Container
        size_frame = tk.Frame(self.left_pane, bg="#1e1e1e")
        size_frame.pack(fill="x", pady=10)
        
        size_hdr = tk.Frame(size_frame, bg="#1e1e1e")
        size_hdr.pack(fill="x")
        tk.Label(size_hdr, text="Size (Sq Ft)", font=("Segoe UI", 11, "bold"), fg="#eeeeee", bg="#1e1e1e").pack(side="left")
        self.size_val_label = tk.Label(size_hdr, text="2,200 sqft", font=("Segoe UI", 11), fg="#3a86f0", bg="#1e1e1e")
        self.size_val_label.pack(side="right")
        
        self.size_scale = tk.Scale(
            size_frame, from_=500, to=4000, orient="horizontal",
            bg="#1e1e1e", fg="#1e1e1e", troughcolor="#2c2c2c",
            activebackground="#3a86f0", highlightthickness=0, bd=0,
            showvalue=0, sliderlength=15, width=12,
            command=self.on_slider_change
        )
        self.size_scale.set(2200)
        self.size_scale.pack(fill="x", pady=(8, 0))
        
        # 2. Bedrooms Slider Container
        bed_frame = tk.Frame(self.left_pane, bg="#1e1e1e")
        bed_frame.pack(fill="x", pady=15)
        
        bed_hdr = tk.Frame(bed_frame, bg="#1e1e1e")
        bed_hdr.pack(fill="x")
        tk.Label(bed_hdr, text="Bedrooms", font=("Segoe UI", 11, "bold"), fg="#eeeeee", bg="#1e1e1e").pack(side="left")
        self.bed_val_label = tk.Label(bed_hdr, text="3 beds", font=("Segoe UI", 11), fg="#3a86f0", bg="#1e1e1e")
        self.bed_val_label.pack(side="right")
        
        self.bed_scale = tk.Scale(
            bed_frame, from_=1, to=5, orient="horizontal",
            bg="#1e1e1e", fg="#1e1e1e", troughcolor="#2c2c2c",
            activebackground="#3a86f0", highlightthickness=0, bd=0,
            showvalue=0, sliderlength=15, width=12,
            command=self.on_slider_change
        )
        self.bed_scale.set(3)
        self.bed_scale.pack(fill="x", pady=(8, 0))
        
        # 3. Neighborhood Score Slider Container
        score_frame = tk.Frame(self.left_pane, bg="#1e1e1e")
        score_frame.pack(fill="x", pady=10)
        
        score_hdr = tk.Frame(score_frame, bg="#1e1e1e")
        score_hdr.pack(fill="x")
        tk.Label(score_hdr, text="Neighborhood Rating", font=("Segoe UI", 11, "bold"), fg="#eeeeee", bg="#1e1e1e").pack(side="left")
        self.score_val_label = tk.Label(score_hdr, text="7.5 / 10", font=("Segoe UI", 11), fg="#3a86f0", bg="#1e1e1e")
        self.score_val_label.pack(side="right")
        
        self.score_scale = tk.Scale(
            score_frame, from_=1.0, to=10.0, orient="horizontal", resolution=0.1,
            bg="#1e1e1e", fg="#1e1e1e", troughcolor="#2c2c2c",
            activebackground="#3a86f0", highlightthickness=0, bd=0,
            showvalue=0, sliderlength=15, width=12,
            command=self.on_slider_change
        )
        self.score_scale.set(7.5)
        self.score_scale.pack(fill="x", pady=(8, 0))
        
        # Separation spacer
        tk.Frame(self.left_pane, height=20, bg="#1e1e1e").pack()
        
        # Estimated Price Display Card
        pred_card = tk.Frame(
            self.left_pane, bg="#0d1f2d", padx=18, pady=18,
            highlightthickness=1, highlightbackground="#1d9bf0", bd=0
        )
        pred_card.pack(fill="x", pady=10)
        
        tk.Label(pred_card, text="ESTIMATED VALUATION", font=("Segoe UI", 9, "bold"), fg="#1d9bf0", bg="#0d1f2d").pack(anchor="w")
        self.price_label = tk.Label(pred_card, text="$0.00", font=("Segoe UI", 26, "bold"), fg="#ffffff", bg="#0d1f2d")
        self.price_label.pack(anchor="w", pady=(5, 0))
        
        # Separation spacer
        tk.Frame(self.left_pane, height=10, bg="#1e1e1e").pack()
        
        # Model Statistics Card
        stats_card = tk.Frame(
            self.left_pane, bg="#262626", padx=18, pady=18,
            highlightthickness=1, highlightbackground="#3d3d3d", bd=0
        )
        stats_card.pack(fill="x", pady=10)
        
        tk.Label(stats_card, text="MODEL METRICS & DETAILS", font=("Segoe UI", 9, "bold"), fg="#aaaaaa", bg="#262626").pack(anchor="w", pady=(0, 8))
        
        metrics_text = f"Accuracy (R²): {r2:.2%}\nMean Error (RMSE): ${rmse:,.2f}"
        tk.Label(stats_card, text=metrics_text, font=("Segoe UI", 10), fg="#ffffff", bg="#262626", justify="left").pack(anchor="w", pady=(0, 12))
        
        tk.Label(stats_card, text="Valuation Formula Weights:", font=("Segoe UI", 9, "bold"), fg="#888888", bg="#262626").pack(anchor="w")
        
        weights_text = (
            f"• Sq Ft: +${unscaled_coefs[0]:.2f} / ft²\n"
            f"• Bedrooms: +${unscaled_coefs[1]:,.2f} / room\n"
            f"• Score: +${unscaled_coefs[2]:,.2f} / point\n"
            f"• Base Value: ${intercept:,.2f}"
        )
        tk.Label(stats_card, text=weights_text, font=("Segoe UI", 9), fg="#cccccc", bg="#262626", justify="left").pack(anchor="w", pady=(5, 0))

    def setup_right_panel(self):
        # Right pane container
        self.right_pane = tk.Frame(self.main_container, bg="#121212")
        self.right_pane.grid(row=0, column=1, sticky="nsew", padx=(15, 0))
        self.right_pane.columnconfigure(0, weight=1)
        self.right_pane.rowconfigure(0, weight=1)
        
        # Use dark background configuration for plots
        plt.style.use('dark_background')
        
        # Create matplotlib figure with 2 subplots side-by-side
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.patch.set_facecolor('#121212')
        self.ax1.set_facecolor('#1e1e1e')
        self.ax2.set_facecolor('#1e1e1e')
        
        # Subplot 1: Price vs. Square Footage
        # Sample 200 random points for rendering performance
        sample_df = df.sample(n=200, random_state=42)
        self.ax1.scatter(sample_df['Size_SqFt'], sample_df['Price'], color='#3a86f0', alpha=0.35, s=15, label='Actual Sales')
        
        # Dynamic lines and markers
        self.reg_line, = self.ax1.plot([], [], color='#00ffcc', linewidth=2.5, label='Model Trend Line')
        self.current_marker, = self.ax1.plot([], [], marker='*', color='#ff3366', markersize=14, markeredgecolor='white', label='Your House')
        
        self.ax1.set_title('Price vs. Square Footage', fontname='Segoe UI', fontsize=12, fontweight='bold', pad=12, color='#ffffff')
        self.ax1.set_xlabel('Square Footage (Sq Ft)', fontsize=9, color='#888888')
        self.ax1.set_ylabel('Price ($)', fontsize=9, color='#888888')
        self.ax1.grid(True, linestyle='--', color='#2d2d2d', alpha=0.5)
        self.ax1.legend(facecolor='#1e1e1e', edgecolor='#2d2d2d', fontsize=8, loc='upper left')
        
        # Subplot 2: Actual vs. Predicted Diagnostic
        test_sample_idx = np.random.choice(len(y_test), size=min(150, len(y_test)), replace=False)
        y_test_sample = y_test.iloc[test_sample_idx]
        y_pred_sample = y_pred[test_sample_idx]
        
        self.ax2.scatter(y_test_sample, y_pred_sample, color='#bd93f9', alpha=0.4, s=15, label='Test Data')
        
        # Draw perfect 45-degree fit reference line
        min_p = min(y_test.min(), y_pred.min())
        max_p = max(y_test.max(), y_pred.max())
        self.ax2.plot([min_p, max_p], [min_p, max_p], color='#ff79c6', linestyle='--', linewidth=1.5, label='Perfect Fit')
        
        # Dynamic crosshair indicators showing where user's house falls on test distribution
        self.current_y_line, = self.ax2.plot([min_p, max_p], [0, 0], color='#ff3366', linestyle=':', alpha=0.7)
        self.current_x_line, = self.ax2.plot([0, 0], [min_p, max_p], color='#ff3366', linestyle=':', alpha=0.7)
        
        self.ax2.set_title('Actual vs. Predicted Valuation', fontname='Segoe UI', fontsize=12, fontweight='bold', pad=12, color='#ffffff')
        self.ax2.set_xlabel('Actual Price ($)', fontsize=9, color='#888888')
        self.ax2.set_ylabel('Predicted Price ($)', fontsize=9, color='#888888')
        self.ax2.grid(True, linestyle='--', color='#2d2d2d', alpha=0.5)
        self.ax2.legend(facecolor='#1e1e1e', edgecolor='#2d2d2d', fontsize=8, loc='upper left')
        
        # Apply clean formatting to ticks
        self.ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))
        self.ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${int(x)/1000:,.0f}k"))
        self.ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${int(x)/1000:,.0f}k"))
        self.ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"${int(x)/1000:,.0f}k"))
        
        # Set bounds matching data range
        self.ax1.set_xlim(500, 4000)
        self.ax1.set_ylim(df['Price'].min() - 40000, df['Price'].max() + 40000)
        self.ax2.set_xlim(df['Price'].min() - 20000, df['Price'].max() + 20000)
        self.ax2.set_ylim(df['Price'].min() - 20000, df['Price'].max() + 20000)
        
        # Embed Figure inside standard Tkinter Frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_pane)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")
        self.canvas_widget.configure(bg="#121212")
        
        self.fig.tight_layout()

    def on_slider_change(self, val):
        # Update live predictions whenever any slider is adjusted
        self.update_prediction()

    def update_prediction(self):
        # Fetch current UI slider values
        size = self.size_scale.get()
        bed = self.bed_scale.get()
        score = self.score_scale.get()
        
        # Update values displayed next to feature titles
        self.size_val_label.configure(text=f"{int(size):,} sqft")
        self.bed_val_label.configure(text=f"{int(bed)} beds" if bed > 1 else "1 bed")
        self.score_val_label.configure(text=f"{score:.1f} / 10")
        
        # Form inputs row for prediction
        input_data = pd.DataFrame([[size, bed, score]], columns=['Size_SqFt', 'Bedrooms', 'Neighborhood_Score'])
        
        # Normalize and predict price
        input_scaled = scaler.transform(input_data)
        pred_price = model.predict(input_scaled)[0]
        
        # Update Estimated Price label on card
        self.price_label.configure(text=f"${pred_price:,.2f}")
        
        # 1. Update dynamic model trend line on Subplot 1
        sizes_grid = np.linspace(500, 4000, 100)
        grid_data = pd.DataFrame({
            'Size_SqFt': sizes_grid,
            'Bedrooms': [bed] * 100,
            'Neighborhood_Score': [score] * 100
        })
        grid_scaled = scaler.transform(grid_data)
        pred_grid = model.predict(grid_scaled)
        
        self.reg_line.set_data(sizes_grid, pred_grid)
        
        # 2. Update dynamic marker position representing user input
        self.current_marker.set_data([size], [pred_price])
        
        # 3. Update evaluation crosshair
        self.current_y_line.set_ydata([pred_price, pred_price])
        self.current_x_line.set_xdata([pred_price, pred_price])
        
        # Trigger lazy canvas redraw
        self.canvas.draw_idle()

# =====================================================================
# STEP 7: APPLICATION INITIATION
# =====================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = HousePredictorApp(root)
    
    # Graceful exit hook to prevent system hang on window destroy
    def on_closing():
        plt.close('all')
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()