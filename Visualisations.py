import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import json
from typing import Dict

class GGobiStyleVisualizer:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_dir = os.path.dirname(config_path)
        self.dataset_name = os.path.splitext(os.path.basename(config_path))[0]

        # Load configuration
        self.config = self._load_config()
        self.target_col = self.config['target_column']
        self.id_col = self.config.get('id_column', 'object_id')

        # Load data
        self.train_df, self.test_df = self._load_data()

        # Visualization settings
        self.output_dir = os.path.join('visualizations', self.dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.animation_steps = 36  # Smooth transitions
        self.point_size = 6
        self.point_opacity = 0.8

    def _load_config(self) -> Dict:
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _load_data(self) -> tuple:
        train_path = os.path.join(self.config_dir, 'Last_training.csv')
        test_path = os.path.join(self.config_dir, 'Last_testing.csv')

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Handle IDs and labels
        if self.id_col not in train_df.columns:
            train_df[self.id_col] = [f"train_{i}" for i in range(len(train_df))]
            test_df[self.id_col] = [f"test_{i}" for i in range(len(test_df))]

        train_df[self.target_col] = train_df[self.target_col].astype(str)
        test_df[self.target_col] = test_df[self.target_col].astype(str)

        return train_df, test_df

    def _get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = [col for col in df.columns if col not in [self.target_col, self.id_col]]
        return StandardScaler().fit_transform(df[feature_cols].values)

    def _create_projection_figure(self, df: pd.DataFrame, set_name: str):
        features = self._get_feature_matrix(df)
        ids = df[self.id_col]
        labels = df[self.target_col]

        # Create projections
        pca = PCA(n_components=3)
        pca_proj = pca.fit_transform(features)

        tsne = TSNE(n_components=3, perplexity=min(30, len(df)-1), random_state=42)
        tsne_proj = tsne.fit_transform(features)

        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=('PCA 3D Projection', 't-SNE 3D Projection')
        )

        # Add projections with proper hover info
        self._add_projection_traces(fig, pca_proj, labels, ids, row=1, col=1, name='PCA')
        self._add_projection_traces(fig, tsne_proj, labels, ids, row=1, col=2, name='t-SNE')

        # Configure scenes for smooth transitions
        self._configure_scenes(fig)

        # Add animation
        self._add_rotation_animation(fig)

        # Update layout
        fig.update_layout(
            title=f"{self.dataset_name} - {set_name} Set",
            height=800,
            showlegend=True,
            hovermode='closest',
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {
                        'label': 'Play Rotation',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 50, 'redraw': True}}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                    }
                ],
                'pad': {'r': 10, 't': 10},
                'showactive': False,
                'type': 'buttons'
            }]
        )

        # Save interactive plot
        fig.write_html(os.path.join(self.output_dir, f'interactive_{set_name}.html'))

        # Create separate feature distribution plot
        self._create_feature_distributions(df, set_name)

    def _add_projection_traces(self, fig, projection, labels, ids, row, col, name):
        unique_labels = sorted(labels.unique())

        for label in unique_labels:
            mask = labels == label
            fig.add_trace(
                go.Scatter3d(
                    x=projection[mask, 0],
                    y=projection[mask, 1],
                    z=projection[mask, 2],
                    mode='markers',
                    name=f'{label}',
                    text=[f"{name}<br>ID: {id}<br>Class: {label}"
                          for id in ids[mask]],
                    hovertemplate='%{text}<extra></extra>',
                    marker=dict(
                        size=self.point_size,
                        opacity=self.point_opacity,
                        line=dict(width=0.5, color='DarkSlateGrey')
                    ),
                    showlegend=True
                ),
                row=row, col=col
            )

    def _configure_scenes(self, fig):
        # Configure both scenes with consistent settings
        scene_settings = dict(
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.1),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            ),
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        )

        fig.update_scenes(scene_settings, row=1, col=1)
        fig.update_scenes(scene_settings, row=1, col=2)

    def _add_rotation_animation(self, fig):
        frames = []

        for i in range(self.animation_steps):
            angle = 2 * np.pi * i / self.animation_steps
            eye_x = np.cos(angle) * 2
            eye_y = np.sin(angle) * 2

            # Create frame with updated camera positions for both scenes
            frame = go.Frame(
                layout=dict(
                    scene=dict(
                        camera=dict(eye=dict(x=eye_x, y=eye_y, z=0.3))
                    ),
                    scene2=dict(
                        camera=dict(eye=dict(x=eye_x, y=eye_y, z=0.3))
                    )
                )
            )
            frames.append(frame)

        fig.frames = frames

    def _create_feature_distributions(self, df: pd.DataFrame, set_name: str):
        feature_cols = [col for col in df.columns if col not in [self.target_col, self.id_col]]
        sample_features = feature_cols[:5]  # Show first 5 features

        fig = go.Figure()

        for feature in sample_features:
            fig.add_trace(
                go.Violin(
                    x=df[self.target_col],
                    y=df[feature],
                    name=feature,
                    box_visible=True,
                    meanline_visible=True,
                    points=False
                )
            )

        fig.update_layout(
            title=f"Feature Distributions by Class - {set_name} Set",
            xaxis_title="Class",
            yaxis_title="Feature Value",
            height=600,
            width=1000
        )

        fig.write_html(os.path.join(self.output_dir, f'feature_distributions_{set_name}.html'))

    def create_visualizations(self):
        print(f"Creating GGobi-style visualizations for {self.dataset_name}...")

        for set_name, df in [('train', self.train_df), ('test', self.test_df)]:
            print(f"Processing {set_name} set ({len(df)} samples)...")
            self._create_projection_figure(df, set_name)
            self._create_correlation_matrix(df, set_name)

        print(f"\nAll visualizations saved to: {self.output_dir}")

    def _create_correlation_matrix(self, df: pd.DataFrame, set_name: str):
        feature_cols = [col for col in df.columns if col not in [self.target_col, self.id_col]]
        corr = df[feature_cols].corr()

        fig = px.imshow(
            corr,
            title=f'Feature Correlation Matrix ({set_name} set)',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect='auto',
            labels=dict(color="Correlation")
        )

        fig.update_layout(
            width=1000,
            height=800,
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )

        fig.write_html(os.path.join(self.output_dir, f'correlation_{set_name}.html'))

def main():
    config_path = input("Enter path to config file: ").strip()
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    try:
        visualizer = GGobiStyleVisualizer(config_path)
        visualizer.create_visualizations()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
