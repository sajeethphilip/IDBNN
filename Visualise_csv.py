import pygame
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys
import json
from pygame.locals import *
'''
    Vector Direction:

        Features pointing in the same direction are positively correlated

        Features pointing in opposite directions are negatively correlated

        Orthogonal features are uncorrelated

    Vector Length:

        Longer vectors indicate features with larger variances

        Shorter vectors indicate features with smaller variances

        The length represents the feature's contribution to the PCA components

    Component Contribution:

        Vectors aligned with the X-axis contribute most to PC1

        Vectors aligned with the Y-axis contribute most to PC2

        Vectors aligned with the Z-axis contribute most to PC3

    Cluster Interpretation:

        Data points clustered along a vector direction are primarily influenced by that feature

        Features pointing toward cluster centers are key discriminators

'''
class Interactive3DVisualizer:
    def __init__(self, data_file, config_file):
        self.data_file = data_file
        self.config_file = config_file
        self.df = None
        self.target_column = None
        self.column_names = None  # Store all column names from config
        self.features = None
        self.labels = None
        self.projected_data = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # PCA model storage
        self.pca_model = None

        # Feature vector visualization
        self.show_feature_vectors = False
        self.feature_vectors = None
        self.feature_names = None
        self.vector_scale = 1.0

        # Axis properties
        self.axis_length = 1.5  # Length of each axis
        self.axis_labels = ['X', 'Y', 'Z']

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

        # Hover information
        self.hover_index = None
        self.hover_threshold = 20  # pixels
        self.metadata_columns = []  # Columns to show in metadata
        self.current_mouse_pos = (0, 0)  # Track mouse position


    def load_config(self):
        """Load dataset configuration including metadata columns"""
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        self.target_column = config['target_column']
        self.column_names = config.get('column_names')
        self.metadata_columns = config.get('metadata_columns', [])
        # Add target to metadata if not already included
        if self.target_column not in self.metadata_columns:
            self.metadata_columns.append(self.target_column)

    def draw_hover_info(self):
        """Display metadata for the point under the mouse cursor"""
        if self.hover_index is None:
            return

        point = self.df.iloc[self.hover_index]
        x, y = self.current_hover_pos

        # Create info box
        box_width = 300
        box_height = 150
        box_x = min(x + 20, self.width - box_width - 10)
        box_y = min(y + 20, self.height - box_height - 10)

        # Draw box background
        pygame.draw.rect(self.screen, (30, 30, 40),
                        (box_x, box_y, box_width, box_height))
        pygame.draw.rect(self.screen, (100, 150, 255),
                        (box_x, box_y, box_width, box_height), 2)

        # Draw point index
        idx_text = self.font.render(f"Index: {self.hover_index}", True, (255, 200, 100))
        self.screen.blit(idx_text, (box_x + 10, box_y + 10))

        # Draw class label
        class_name = self.label_encoder.inverse_transform([self.labels[self.hover_index]])[0]
        class_text = self.font.render(f"Class: {class_name}", True, self.colors[self.labels[self.hover_index]])
        self.screen.blit(class_text, (box_x + 10, box_y + 35))

        # Draw metadata
        y_offset = 60
        for col in self.metadata_columns[:3]:  # Show up to 3 metadata columns
            if col in self.df.columns:
                value = point[col]
                if isinstance(value, float):
                    value = f"{value:.4f}"
                text = self.font.render(f"{col}: {value}", True, (200, 200, 200))
                self.screen.blit(text, (box_x + 10, box_y + y_offset))
                y_offset += 20

    def load_data(self):
        """Load CSV data using config to select columns"""
        # Read only specified columns if available
        if self.column_names:
            # Ensure target column is included
            if self.target_column not in self.column_names:
                self.column_names.append(self.target_column)

            self.df = pd.read_csv(self.data_file, usecols=self.column_names)
            print(f"Using {len(self.column_names)} columns from config")
        else:
            self.df = pd.read_csv(self.data_file)
            print("Using all columns from CSV")

        print(f"Columns in dataset: {list(self.df.columns)}")

        # Verify target column exists
        if self.target_column not in self.df.columns:
            print(f"\nError: Target column '{self.target_column}' not found in dataset.")
            sys.exit(1)

        # Create feature list (all columns except target)
        feature_columns = [col for col in self.df.columns if col != self.target_column]

        # Handle case where column_names included target
        if self.target_column in feature_columns:
            feature_columns.remove(self.target_column)

        self.features = self.df[feature_columns]
        self.labels = self.df[self.target_column]
        print(f"Using {len(feature_columns)} feature columns")

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

    def draw_axes(self):
        """Draw 3D axes that rotate with the view"""
        # Define axis endpoints in 3D space
        axes = [
            (self.axis_length, 0, 0),  # X-axis
            (0, self.axis_length, 0),  # Y-axis
            (0, 0, self.axis_length)   # Z-axis
        ]

        axis_colors = [
            (255, 50, 50),    # Red for X
            (50, 255, 50),    # Green for Y
            (50, 100, 255)    # Blue for Z
        ]

        # Draw origin point
        origin_rot = self.rotate_point([0, 0, 0], self.rotation_x, self.rotation_y)
        origin_2d, origin_size = self.project_to_2d(origin_rot)
        pygame.draw.circle(self.screen, (200, 200, 200),
                          (int(origin_2d[0]), int(origin_2d[1])),
                          max(2, origin_size))

        # Draw each axis
        for i, axis in enumerate(axes):
            # Rotate and project axis endpoint
            axis_rot = self.rotate_point(axis, self.rotation_x, self.rotation_y)
            axis_2d, axis_size = self.project_to_2d(axis_rot)

            # Only draw if in front of camera
            if axis_rot[2] > -4:
                # Draw axis line
                pygame.draw.line(self.screen, axis_colors[i],
                                (int(origin_2d[0]), int(origin_2d[1])),
                                (int(axis_2d[0]), int(axis_2d[1])),
                                2)

                # Draw axis label
                offset_x = 10 if axis_2d[0] > origin_2d[0] else -30
                offset_y = 10 if axis_2d[1] > origin_2d[1] else -30

                label = self.font.render(self.axis_labels[i], True, axis_colors[i])
                self.screen.blit(label, (axis_2d[0] + offset_x, axis_2d[1] + offset_y))

    def project_data(self):
        """Project data to 3D using PCA"""
        pca = PCA(n_components=3)
        self.projected_data = pca.fit_transform(self.features)
        self.pca_model = pca  # Store PCA model for reverse mapping

        # Create feature vectors
        self.create_feature_vectors()

        # Normalize to [-1, 1] range
        min_vals = self.projected_data.min(axis=0)
        max_vals = self.projected_data.max(axis=0)
        self.projected_data = 2 * (self.projected_data - min_vals) / (max_vals - min_vals) - 1

        print(f"Data projected to 3D with shape: {self.projected_data.shape}")

    def create_feature_vectors(self):
        """Create vectors representing original features in PCA space"""
        # Get feature names from DataFrame
        if hasattr(self.features, 'columns'):
            self.feature_names = self.features.columns.tolist()
        else:
            self.feature_names = [f"Feature {i}" for i in range(self.features.shape[1])]

        # Create unit vectors for each feature in original space
        identity_matrix = np.eye(self.features.shape[1])

        # Transform to PCA space
        self.feature_vectors = self.pca_model.transform(identity_matrix)

        # Normalize vectors for consistent scaling
        norms = np.linalg.norm(self.feature_vectors, axis=1)
        self.feature_vectors = self.feature_vectors / norms[:, np.newaxis] * 0.5

    def draw_feature_vectors(self):
        """Draw vectors representing original features in PCA space"""
        if self.feature_vectors is None or not self.show_feature_vectors or len(self.feature_vectors) == 0:
            return

        # Get origin point
        origin = np.array([0, 0, 0])
        origin_rot = self.rotate_point(origin, self.rotation_x, self.rotation_y)
        origin_2d, _ = self.project_to_2d(origin_rot)

        # Draw each feature vector
        for i, vec in enumerate(self.feature_vectors):
            # Scale vector
            scaled_vec = vec * self.vector_scale

            # Rotate and project vector endpoint
            vec_rot = self.rotate_point(scaled_vec, self.rotation_x, self.rotation_y)
            vec_2d, size = self.project_to_2d(vec_rot)

            # Only draw if in front of camera
            if vec_rot[2] > -4:
                # Draw vector line
                pygame.draw.line(self.screen, (255, 255, 100),
                                (int(origin_2d[0]), int(origin_2d[1])),
                                (int(vec_2d[0]), int(vec_2d[1])),
                                2)

                # Draw feature name
                name = self.feature_names[i]
                label = self.font.render(name, True, (255, 255, 200))
                self.screen.blit(label, (int(vec_2d[0] + 5), int(vec_2d[1] - 10)))

                # Draw vector head
                pygame.draw.circle(self.screen, (255, 200, 50),
                                  (int(vec_2d[0]), int(vec_2d[1])), 4)


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

        # Add feature vector info if enabled
        if self.show_feature_vectors:
            vec_text = f"Feature Vectors: ON (Scale: {self.vector_scale:.1f})"
            text_surf = self.font.render(vec_text, True, (255, 255, 100))
            self.screen.blit(text_surf, (20, 230))

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
                    elif event.key == K_v:  # Toggle feature vectors
                        self.show_feature_vectors = not self.show_feature_vectors
                    elif event.key == K_PLUS or event.key == K_EQUALS:  # Increase vector scale
                        self.vector_scale *= 1.2
                    elif event.key == K_MINUS:  # Decrease vector scale
                        self.vector_scale /= 1.2

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
                else:
                    # Find closest point to mouse
                    self.hover_index = None
                    min_distance = float('inf')

            # Clear screen
            self.screen.fill((20, 20, 30))

            # Find closest point to mouse
            self.hover_index = None
            min_distance = float('inf')


            # Draw data points
            for i, point in enumerate(self.projected_data):
                rotated = self.rotate_point(point, self.rotation_x, self.rotation_y)
                # Only consider points in front of camera
                if rotated[2] <= -4:
                    continue


                # Only draw points in front of camera
                if rotated[2] > -4:
                    pos_2d, size = self.project_to_2d(rotated)
                    color = self.colors[self.labels[i]]
                    pygame.draw.circle(self.screen, color, (int(pos_2d[0]), int(pos_2d[1])), size)
                    dx = pos_2d[0] - self.current_mouse_pos[0]
                    dy = pos_2d[1] - self.current_mouse_pos[1]
                    distance = dx*dx + dy*dy

                    if distance < min_distance and distance < self.hover_threshold**2:
                        min_distance = distance
                        self.hover_index = i
                        self.current_hover_pos = pos_2d

            # Draw feature vectors
            self.draw_feature_vectors()

            # Draw 3D axes
            self.draw_axes()

            # Draw UI elements
            self.draw_info_panel()
            self.draw_legend()

            # Draw hover information
            self.draw_hover_info()

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
