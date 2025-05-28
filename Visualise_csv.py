import pygame
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys
import json
from pygame.locals import *

class Interactive3DVisualizer:
    def __init__(self, data_file, config_file):
        self.data_file = data_file
        self.config_file = config_file
        self.df = None
        self.target_column = None
        self.feature_columns = None  # New: store feature columns from config
        self.features = None
        self.labels = None
        self.projected_data = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # PyGame setup
        pygame.init()
        self.width, self.height = 1200, 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("3D Data Visualization")
        self.font = pygame.font.SysFont('Arial', 16)
        self.clock = pygame.time.Clock()

        # Camera controls
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = 1.0
        self.translation = [0, 0]
        self.dragging = False
        self.last_mouse_pos = (0, 0)

        # Load data
        self.load_config()
        self.load_data()
        self.preprocess()
        self.project_data()

    def load_config(self):
        """Load dataset configuration to identify target and feature columns"""
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        self.target_column = config['target_column']
        # Get feature columns if specified in config
        self.feature_columns = config.get('features')  # Returns None if not found

    def load_data(self):
        """Load CSV data and separate features from target using config"""
        self.df = pd.read_csv(self.data_file)
        print(f"Columns in dataset: {list(self.df.columns)}")

        # Verify target column exists
        if self.target_column not in self.df.columns:
            print(f"\nError: Target column '{self.target_column}' not found in dataset.")
            sys.exit(1)

        # Use feature columns if specified in config
        if self.feature_columns:
            # Verify all feature columns exist
            missing = [col for col in self.feature_columns if col not in self.df.columns]
            if missing:
                print(f"\nError: Feature columns not found in dataset: {missing}")
                sys.exit(1)

            # Verify target isn't included in features
            if self.target_column in self.feature_columns:
                print(f"\nNote: Target column '{self.target_column}' found in feature list. Removing it.")
                self.feature_columns = [col for col in self.feature_columns if col != self.target_column]

            self.features = self.df[self.feature_columns]
            self.labels = self.df[self.target_column]
            print(f"Using {len(self.feature_columns)} feature columns from config")
        else:
            # Use all columns except target as features
            self.features = self.df.drop(columns=[self.target_column])
            self.labels = self.df[self.target_column]
            print(f"Using all columns except target ({self.target_column}) as features")

    def preprocess(self):
        """Standardize features and encode labels"""
        # Standardize features
        self.features = self.scaler.fit_transform(self.features)

        # Encode labels
        self.labels = self.label_encoder.fit_transform(self.labels)

        # Create color mapping
        unique_labels = np.unique(self.labels)
        self.colors = {}
        for label in unique_labels:
            hue = label / len(unique_labels)
            self.colors[label] = (
                int(255 * (1 - hue)),
                int(255 * hue),
                int(255 * (0.5 + 0.5 * (label % 2)))
            )

    def project_data(self):
        """Project data to 3D using PCA"""
        pca = PCA(n_components=3)
        self.projected_data = pca.fit_transform(self.features)

        # Normalize to [-1, 1] range
        min_vals = self.projected_data.min(axis=0)
        max_vals = self.projected_data.max(axis=0)
        self.projected_data = 2 * (self.projected_data - min_vals) / (max_vals - min_vals) - 1

        print(f"Data projected to 3D with shape: {self.projected_data.shape}")

    def rotate_point(self, point, rx, ry):
        """Rotate point around X and Y axes"""
        # Rotate around X axis
        y = point[1] * np.cos(rx) - point[2] * np.sin(rx)
        z = point[1] * np.sin(rx) + point[2] * np.cos(rx)

        # Rotate around Y axis
        x = point[0] * np.cos(ry) + z * np.sin(ry)
        z = -point[0] * np.sin(ry) + z * np.cos(ry)

        return [x, y, z]

    def project_to_2d(self, point):
        """Project 3D point to 2D screen coordinates"""
        # Apply perspective projection
        z = point[2] + 5  # Distance from camera
        factor = 200 * self.zoom / z
        x = self.width/2 + point[0] * factor + self.translation[0]
        y = self.height/2 - point[1] * factor + self.translation[1]
        size = max(1, int(20 / z))  # Size based on distance

        return (x, y), size

    def draw_info_panel(self):
        """Draw information panel showing controls and stats"""
        # Draw background panel
        pygame.draw.rect(self.screen, (40, 40, 50), (10, 10, 300, 180))

        # Draw title
        title = self.font.render(f"3D Data Visualization: {os.path.basename(self.data_file)}", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))

        # Draw controls
        controls = [
            "CONTROLS:",
            "Arrow Keys - Rotate view",
            "Mouse Drag - Pan view",
            "Mouse Wheel - Zoom in/out",
            "R - Reset view",
            "C - Change color mapping",
            "ESC - Quit"
        ]

        for i, text in enumerate(controls):
            text_surf = self.font.render(text, True, (200, 200, 200))
            self.screen.blit(text_surf, (20, 50 + i*20))

        # Draw stats
        stats = [
            f"Points: {len(self.projected_data)}",
            f"Features: {self.features.shape[1]}",
            f"Classes: {len(np.unique(self.labels))}",
            f"Rotation: X:{int(np.degrees(self.rotation_x))}° Y:{int(np.degrees(self.rotation_y))}°"
        ]

        for i, text in enumerate(stats):
            text_surf = self.font.render(text, True, (200, 200, 255))
            self.screen.blit(text_surf, (20, 150 + i*20))

    def draw_legend(self):
        """Draw legend showing class colors"""
        unique_labels = np.unique(self.labels)
        class_names = self.label_encoder.inverse_transform(unique_labels)

        # Draw legend background
        pygame.draw.rect(self.screen, (40, 40, 50), (self.width - 210, 10, 200, 40 + len(unique_labels)*25))

        # Draw title
        title = self.font.render("CLASS LEGEND", True, (255, 255, 255))
        self.screen.blit(title, (self.width - 200, 20))

        # Draw class items
        for i, (label, name) in enumerate(zip(unique_labels, class_names)):
            color = self.colors[label]
            pygame.draw.circle(self.screen, color, (self.width - 190, 60 + i*25), 8)
            text = self.font.render(name, True, (200, 200, 200))
            self.screen.blit(text, (self.width - 170, 55 + i*25))

    def run(self):
        """Main visualization loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

                # Handle keyboard input
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key == K_UP:
                        self.rotation_x += 0.1
                    elif event.key == K_DOWN:
                        self.rotation_x -= 0.1
                    elif event.key == K_LEFT:
                        self.rotation_y += 0.1
                    elif event.key == K_RIGHT:
                        self.rotation_y -= 0.1
                    elif event.key == K_r:  # Reset view
                        self.rotation_x = 0
                        self.rotation_y = 0
                        self.zoom = 1.0
                        self.translation = [0, 0]
                    elif event.key == K_c:  # Change color mapping
                        self.preprocess()

                # Handle mouse input
                elif event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        self.dragging = True
                        self.last_mouse_pos = event.pos
                    elif event.button == 4:  # Mouse wheel up
                        self.zoom *= 1.1
                    elif event.button == 5:  # Mouse wheel down
                        self.zoom /= 1.1

                elif event.type == MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False

                elif event.type == MOUSEMOTION:
                    if self.dragging:
                        dx = event.pos[0] - self.last_mouse_pos[0]
                        dy = event.pos[1] - self.last_mouse_pos[1]
                        self.translation[0] += dx
                        self.translation[1] += dy
                        self.last_mouse_pos = event.pos

            # Clear screen
            self.screen.fill((20, 20, 30))

            # Draw data points
            for i, point in enumerate(self.projected_data):
                rotated = self.rotate_point(point, self.rotation_x, self.rotation_y)
                pos_2d, size = self.project_to_2d(rotated)

                # Only draw points in front of camera
                if rotated[2] > -4:
                    color = self.colors[self.labels[i]]
                    pygame.draw.circle(self.screen, color, (int(pos_2d[0]), int(pos_2d[1])), size)

            # Draw UI elements
            self.draw_info_panel()
            self.draw_legend()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_3d.py <data_csv> <config_json>")
        sys.exit(1)

    data_file = sys.argv[1]
    config_file = sys.argv[2]

    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        sys.exit(1)

    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        sys.exit(1)

    visualizer = Interactive3DVisualizer(data_file, config_file)
    visualizer.run()
