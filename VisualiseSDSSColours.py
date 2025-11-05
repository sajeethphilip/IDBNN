import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Download SDSS data for stars and QSOs
def download_sdss_data():
    """
    Download SDSS data for stars and QSOs
    We'll use the SDSS DR16 catalog via their API
    """
    print("Downloading SDSS data...")

    # Base URL for SDSS DR16 API
    base_url = "https://skyserver.sdss.org/dr16/en/tools/search/x_sql.aspx"

    # Query for stars (using model magnitudes which are more reliable)
    star_query = """
    SELECT TOP 2000
        p.ra, p.dec, p.u, p.g, p.r, p.i, p.z,
        p.modelMag_u, p.modelMag_g, p.modelMag_r, p.modelMag_i, p.modelMag_z
    FROM PhotoPrimary p
    WHERE
        p.type = 6  -- Stars
        AND p.u BETWEEN 14 AND 22
        AND p.g BETWEEN 14 AND 22
        AND p.u - p.g > -0.5
        AND p.u - p.g < 2.5
    """

    # Query for QSOs with measured redshifts
    qso_query = """
    SELECT TOP 2000
        p.ra, p.dec, p.u, p.g, p.r, p.i, p.z,
        p.modelMag_u, p.modelMag_g, p.modelMag_r, p.modelMag_i, p.modelMag_z,
        s.z as redshift
    FROM PhotoPrimary p
    JOIN SpecObj s ON p.objID = s.bestObjID
    WHERE
        s.class = 'QSO'
        AND s.z > 0
        AND s.z < 4
        AND p.u BETWEEN 14 AND 22
        AND p.g BETWEEN 14 AND 22
    """

    def execute_query(query, object_type):
        """Execute SQL query against SDSS database"""
        try:
            print(f"Downloading {object_type} data...")
            response = requests.get(base_url, params={'cmd': query, 'format': 'csv'})
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))
                print(f"Successfully downloaded {len(df)} {object_type}")
                print(f"Columns: {df.columns.tolist()}")
                return df
            else:
                print(f"Query failed with status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    # Try to download real data
    stars_df = execute_query(star_query, "stars")
    qso_df = execute_query(qso_query, "QSOs")

    # If download fails, use sample data
    if stars_df is None or qso_df is None:
        print("Download failed, using sample data...")
        return create_sample_data()

    # Process the downloaded data - handle different column naming conventions
    def process_sdss_data(df, is_qso=False):
        """Process SDSS data to extract colors"""
        print(f"Processing data with columns: {df.columns.tolist()}")

        # Check available magnitude columns and use what's available
        mag_columns = {}
        for band in ['u', 'g', 'r', 'i', 'z']:
            # Try different possible column names
            possible_names = [
                f'modelMag_{band}',
                f'psfMag_{band}',
                f'{band}',
                f'modelMag{band}',
                f'psfMag{band}'
            ]
            for name in possible_names:
                if name in df.columns:
                    mag_columns[band] = name
                    break
            if band not in mag_columns:
                print(f"Warning: Could not find magnitude column for band {band}")

        # Create colors DataFrame
        colors_data = {}
        if 'u' in mag_columns and 'g' in mag_columns:
            colors_data['u-g'] = df[mag_columns['u']] - df[mag_columns['g']]
        if 'g' in mag_columns and 'r' in mag_columns:
            colors_data['g-r'] = df[mag_columns['g']] - df[mag_columns['r']]
        if 'r' in mag_columns and 'i' in mag_columns:
            colors_data['r-i'] = df[mag_columns['r']] - df[mag_columns['i']]
        if 'i' in mag_columns and 'z' in mag_columns:
            colors_data['i-z'] = df[mag_columns['i']] - df[mag_columns['z']]

        colors_df = pd.DataFrame(colors_data)

        # Add redshift for QSOs
        if is_qso and 'redshift' in df.columns:
            colors_df['redshift'] = df['redshift']

        # Clean the data - remove outliers and NaN values
        mask = pd.Series(True, index=colors_df.index)
        for col in colors_df.columns:
            if col != 'redshift':
                col_mask = (colors_df[col] > -1) & (colors_df[col] < 3) & colors_df[col].notna()
                mask = mask & col_mask

        cleaned_df = colors_df[mask].copy()
        print(f"Cleaned data: {len(cleaned_df)} objects remaining")
        return cleaned_df

    try:
        processed_stars = process_sdss_data(stars_df)
        processed_qso = process_sdss_data(qso_df, is_qso=True)

        if len(processed_stars) < 100 or len(processed_qso) < 100:
            print("Not enough clean data, using sample data...")
            return create_sample_data()

        return processed_stars, processed_qso

    except Exception as e:
        print(f"Error processing SDSS data: {e}")
        print("Using sample data instead...")
        return create_sample_data()

def create_sample_data():
    """Create realistic sample data"""
    print("Generating realistic sample data...")
    np.random.seed(42)

    # Generate realistic star colors (SDSS stellar locus)
    n_stars = 1500

    # Main sequence stars follow a specific track in color space
    star_u_g = np.random.normal(1.0, 0.3, n_stars)
    star_g_r = 0.5 * star_u_g + np.random.normal(0, 0.15, n_stars) - 0.2
    star_r_i = 0.4 * star_g_r + np.random.normal(0, 0.1, n_stars) + 0.15
    star_i_z = 0.3 * star_r_i + np.random.normal(0, 0.08, n_stars) + 0.08

    stars_df = pd.DataFrame({
        'u-g': np.clip(star_u_g, -0.5, 2.5),
        'g-r': np.clip(star_g_r, -0.5, 1.5),
        'r-i': np.clip(star_r_i, -0.5, 1.0),
        'i-z': np.clip(star_i_z, -0.5, 0.8)
    })

    # Generate QSO colors at different redshifts
    n_qso = 1200
    redshifts = np.random.uniform(0, 4, n_qso)

    def qso_color_model(z):
        # Realistic QSO color evolution
        # Low-z QSOs are typically bluer than stars
        if z < 0.5:
            u_g = np.random.normal(0.2, 0.2)
            g_r = np.random.normal(0.1, 0.15)
        elif z < 1.5:
            # Colors get redder as Balmer break moves through filters
            u_g = 0.3 + 0.3 * z + np.random.normal(0, 0.2)
            g_r = 0.15 + 0.1 * z + np.random.normal(0, 0.15)
        elif z < 2.5:
            # Lyman-alpha enters u-band, making u-g very red
            u_g = 0.8 + 0.8 * (z - 1.5) + np.random.normal(0, 0.3)
            g_r = 0.3 + 0.2 * (z - 1.5) + np.random.normal(0, 0.2)
        else:
            # Very high redshift, Lyman-alpha in g-band
            u_g = 2.0 + 0.5 * (z - 2.5) + np.random.normal(0, 0.4)
            g_r = 0.6 + 0.3 * (z - 2.5) + np.random.normal(0, 0.25)

        r_i = 0.1 + 0.05 * z + np.random.normal(0, 0.1)
        i_z = 0.05 + 0.03 * z + np.random.normal(0, 0.08)

        return (
            np.clip(u_g, -0.5, 3.0),
            np.clip(g_r, -0.5, 2.0),
            np.clip(r_i, -0.5, 1.0),
            np.clip(i_z, -0.5, 0.8)
        )

    qso_colors = np.array([qso_color_model(z) for z in redshifts])

    qso_df = pd.DataFrame({
        'u-g': qso_colors[:, 0],
        'g-r': qso_colors[:, 1],
        'r-i': qso_colors[:, 2],
        'i-z': qso_colors[:, 3],
        'redshift': redshifts
    })

    print(f"Generated {len(stars_df)} stars and {len(qso_df)} QSOs")
    return stars_df, qso_df

def apply_redshift_to_qso(qso_df, target_z):
    """
    Apply redshift effects to QSO colors
    """
    if not isinstance(qso_df, pd.DataFrame):
        raise ValueError("qso_df should be a pandas DataFrame")

    # For sample data with redshift column, use individual redshifts
    # For template QSOs, simulate evolution from z=0
    simulated_colors = []

    for _, qso in qso_df.iterrows():
        if 'redshift' in qso_df.columns:
            original_z = qso['redshift']
            # For real QSOs, we're showing how they would appear at different z
            z_effect = target_z - original_z
        else:
            # For template QSOs, simulate evolution from z=0 to target_z
            original_z = 0
            z_effect = target_z

        # Color evolution model
        if target_z > 2.0:
            u_g_shift = 0.7 * z_effect
        else:
            u_g_shift = 0.3 * z_effect

        g_r_shift = 0.2 * z_effect
        r_i_shift = 0.1 * z_effect
        i_z_shift = 0.05 * z_effect

        u_g = qso['u-g'] + u_g_shift + np.random.normal(0, 0.15)
        g_r = qso['g-r'] + g_r_shift + np.random.normal(0, 0.1)
        r_i = qso['r-i'] + r_i_shift + np.random.normal(0, 0.08)
        i_z = qso['i-z'] + i_z_shift + np.random.normal(0, 0.06)

        simulated_colors.append([
            np.clip(u_g, -0.5, 3.0),
            np.clip(g_r, -0.5, 2.0),
            np.clip(r_i, -0.5, 1.0),
            np.clip(i_z, -0.5, 0.8)
        ])

    simulated_df = pd.DataFrame(simulated_colors, columns=['u-g', 'g-r', 'r-i', 'i-z'])
    return simulated_df

def create_interactive_visualization(stars_df, qso_df):
    """Create interactive Plotly visualization"""

    print("Creating interactive visualization...")

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('u-g vs g-r', 'g-r vs r-i', 'r-i vs i-z', 'u-g vs r-i'),
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    # Initial plot at z=0
    initial_z = 0.0
    simulated_qso = apply_redshift_to_qso(qso_df, initial_z)

    # Create traces
    star_trace1 = go.Scatter(
        x=stars_df['u-g'], y=stars_df['g-r'],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.5),
        name='Stars (z=0)',
        showlegend=True
    )

    star_trace2 = go.Scatter(
        x=stars_df['g-r'], y=stars_df['r-i'],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.5),
        showlegend=False
    )

    star_trace3 = go.Scatter(
        x=stars_df['r-i'], y=stars_df['i-z'],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.5),
        showlegend=False
    )

    star_trace4 = go.Scatter(
        x=stars_df['u-g'], y=stars_df['r-i'],
        mode='markers',
        marker=dict(size=3, color='blue', opacity=0.5),
        showlegend=False
    )

    qso_trace1 = go.Scatter(
        x=simulated_qso['u-g'], y=simulated_qso['g-r'],
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.6),
        name=f'QSOs (z={initial_z:.1f})',
        showlegend=True
    )

    qso_trace2 = go.Scatter(
        x=simulated_qso['g-r'], y=simulated_qso['r-i'],
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.6),
        showlegend=False
    )

    qso_trace3 = go.Scatter(
        x=simulated_qso['r-i'], y=simulated_qso['i-z'],
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.6),
        showlegend=False
    )

    qso_trace4 = go.Scatter(
        x=simulated_qso['u-g'], y=simulated_qso['r-i'],
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.6),
        showlegend=False
    )

    # Add traces to subplots
    fig.add_trace(star_trace1, row=1, col=1)
    fig.add_trace(qso_trace1, row=1, col=1)

    fig.add_trace(star_trace2, row=1, col=2)
    fig.add_trace(qso_trace2, row=1, col=2)

    fig.add_trace(star_trace3, row=2, col=1)
    fig.add_trace(qso_trace3, row=2, col=1)

    fig.add_trace(star_trace4, row=2, col=2)
    fig.add_trace(qso_trace4, row=2, col=2)

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'SDSS Color-Color Diagrams: QSO Redshift Evolution (z={initial_z:.1f})',
            x=0.5,
            font=dict(size=20)
        ),
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # Update axes
    fig.update_xaxes(title_text='u - g', row=1, col=1, range=[-0.5, 2.5])
    fig.update_yaxes(title_text='g - r', row=1, col=1, range=[-0.5, 1.5])

    fig.update_xaxes(title_text='g - r', row=1, col=2, range=[-0.5, 1.5])
    fig.update_yaxes(title_text='r - i', row=1, col=2, range=[-0.5, 1.0])

    fig.update_xaxes(title_text='r - i', row=2, col=1, range=[-0.5, 1.0])
    fig.update_yaxes(title_text='i - z', row=2, col=1, range=[-0.5, 0.8])

    fig.update_xaxes(title_text='u - g', row=2, col=2, range=[-0.5, 2.5])
    fig.update_yaxes(title_text='r - i', row=2, col=2, range=[-0.5, 1.0])

    # Add slider
    steps = []
    z_values = np.linspace(0, 4, 21)

    for i, z in enumerate(z_values):
        simulated_qso_step = apply_redshift_to_qso(qso_df, z)

        step = dict(
            method='update',
            args=[
                {
                    'x': [
                        stars_df['u-g'], simulated_qso_step['u-g'],
                        stars_df['g-r'], simulated_qso_step['g-r'],
                        stars_df['r-i'], simulated_qso_step['r-i'],
                        stars_df['u-g'], simulated_qso_step['u-g']
                    ],
                    'y': [
                        stars_df['g-r'], simulated_qso_step['g-r'],
                        stars_df['r-i'], simulated_qso_step['r-i'],
                        stars_df['i-z'], simulated_qso_step['i-z'],
                        stars_df['r-i'], simulated_qso_step['r-i']
                    ]
                },
                {'title': f'SDSS Color-Color Diagrams: QSO Redshift Evolution (z={z:.1f})'}
            ],
            label=f'z={z:.1f}'
        )
        steps.append(step)

    sliders = [dict(active=0, currentvalue={"prefix": "Redshift: "}, pad={"t": 50}, steps=steps)]
    fig.update_layout(sliders=sliders)

    # Add explanation
    fig.add_annotation(
        text="Drag the slider to see how QSO colors evolve with redshift and merge with stellar colors",
        xref="paper", yref="paper", x=0.5, y=-0.12,
        showarrow=False, font=dict(size=12), xanchor='center'
    )

    return fig

def create_static_comparison(stars_df, qso_df):
    """Create static comparison plots"""
    print("Creating static comparison plot...")

    key_redshifts = [0.0, 1.0, 2.0, 3.0, 4.0]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, z in enumerate(key_redshifts):
        if i >= len(axes): break

        simulated_qso = apply_redshift_to_qso(qso_df, z)

        axes[i].scatter(stars_df['u-g'], stars_df['g-r'],
                       c='blue', alpha=0.5, s=2, label='Stars (z=0)')
        axes[i].scatter(simulated_qso['u-g'], simulated_qso['g-r'],
                       c='red', alpha=0.6, s=3, label=f'QSOs (z={z})')

        axes[i].set_xlabel('u - g')
        axes[i].set_ylabel('g - r')
        axes[i].set_title(f'Redshift z = {z}')
        axes[i].set_xlim(-0.5, 2.5)
        axes[i].set_ylim(-0.5, 1.5)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=10)

    # Add explanation
    axes[-1].axis('off')
    explanation = """
    SDSS Color-Color Diagram Evolution

    • Blue: Main sequence stars (z=0)
    • Red: QSOs at different redshifts

    Key Effects:
    1. Low-z QSOs: Bluer than stars
    2. z ~ 0.5-2.0: Overlap with stellar locus
    3. z > 2.0: Lyman-alpha makes u-g very red
    4. z > 3.0: Lyman-alpha in g-band

    This shows why photometric classification
    is challenging - QSOs and stars overlap
    in color space at certain redshifts!
    """
    axes[-1].text(0.1, 0.9, explanation, transform=axes[-1].transAxes,
                 fontsize=12, va='top', ha='left', linespacing=1.5)

    plt.tight_layout()
    plt.savefig('sdss_redshift_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    print("SDSS Redshift Color Evolution Visualizer")
    print("=" * 50)

    stars_df, qso_df = download_sdss_data()
    print(f"Final data: {len(stars_df)} stars, {len(qso_df)} QSOs")

    fig = create_interactive_visualization(stars_df, qso_df)
    fig.write_html("sdss_redshift_evolution.html")
    print("✓ Interactive visualization saved as 'sdss_redshift_evolution.html'")

    create_static_comparison(stars_df, qso_df)
    print("✓ Static comparison saved as 'sdss_redshift_comparison.png'")

    print("\n" + "=" * 50)
    print("SUCCESS! Open 'sdss_redshift_evolution.html' in your browser.")
    print("Features:")
    print("- Drag slider to change QSO redshift (0 to 4)")
    print("- Watch QSO colors evolve and merge with stars")
    print("- Four different color-color diagrams")
