#!/usr/bin/env python3
"""
BSG_Integrated_Development_Environment.py
An integrated development environment for BeamerSlideGenerator
Combines GUI editing, syntax highlighting, and presentation generation.

"""
import os
import sys
from pathlib import Path

def setup_package_paths():
    """Setup Python paths for both source and installed runs"""
    try:
        # Get the directory where this file is located
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent

        # Check if we're in source directory (look for key files)
        is_source = (current_dir / 'BeamerSlideGenerator.py').exists()

        if is_source:
            # Running from source - add current directory to path
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            print(f"Running from source: {current_dir}")
            return current_dir, current_dir / 'resources'

        # Check if we're in a pip installed package
        try:
            import bsg_ide
            package_dir = Path(bsg_ide.__file__).parent
            if str(package_dir) not in sys.path:
                sys.path.insert(0, str(package_dir))
            resources_dir = package_dir / 'resources'
            if not resources_dir.exists():
                resources_dir = package_dir
            print(f"Running from installed package: {package_dir}")
            return package_dir, resources_dir
        except ImportError:
            pass

        # Check common installation locations
        common_locations = [
            Path.home() / '.local/lib/bsg-ide',
            Path.home() / '.local/share/bsg-ide',
            Path('/usr/local/share/bsg-ide'),
            Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages' / 'bsg_ide',
        ]

        for loc in common_locations:
            if (loc / 'BeamerSlideGenerator.py').exists():
                if str(loc) not in sys.path:
                    sys.path.insert(0, str(loc))
                resources_dir = loc / 'resources'
                if not resources_dir.exists():
                    resources_dir = loc
                print(f"Found installation at: {loc}")
                return loc, resources_dir

        # Fallback to current directory
        print(f"Using current directory as fallback: {current_dir}")
        return current_dir, current_dir / 'resources'

    except Exception as e:
        print(f"Warning: Error setting up paths: {e}")
        return Path.cwd(), Path.cwd() / 'resources'

# Run path setup immediately
PACKAGE_ROOT, RESOURCES_DIR = setup_package_paths()

# Now try to import BeamerSlideGenerator
try:
    from BeamerSlideGenerator import (
        get_beamer_preamble,
        process_media,
        generate_latex_code,
        download_youtube_video,
        construct_search_query,
        open_google_image_search,
        process_input_file,
        set_terminal_io,
        convert_media,
        download_media
    )
    print("✓ Successfully imported BeamerSlideGenerator")
except ImportError as e:
    print(f"Error importing BeamerSlideGenerator: {e}")
    print(f"Python path: {sys.path}")
    print(f"Looking for BeamerSlideGenerator.py in: {PACKAGE_ROOT}")

    # Try alternative import
    try:
        from bsg_ide.BeamerSlideGenerator import (
            get_beamer_preamble,
            process_media,
            generate_latex_code,
            download_youtube_video,
            construct_search_query,
            open_google_image_search,
            process_input_file,
            set_terminal_io,
            convert_media,
            download_media
        )
        print("✓ Successfully imported BeamerSlideGenerator via bsg_ide")
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        print("\n" + "="*60)
        print("ERROR: BeamerSlideGenerator module not found!")
        print("="*60)
        print("\nPlease ensure one of the following:")
        print("1. BeamerSlideGenerator.py is in the same directory as BSG_IDE.py")
        print("2. The package is properly installed via: pip install bsg-ide")
        print("3. Reinstall the package: pip install --force-reinstall bsg-ide")
        print("\nCurrent Python path:")
        for p in sys.path:
            print(f"  {p}")
        sys.exit(1)

#------------------------------Check and install ----------------------------------------------
import os,re
import sys
import tempfile
import subprocess
#------------------------------------------------------------------------------------------------------
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta
import venv
import platform
from pathlib import Path
import site
import socket
from importlib import util
global working_folder
working_folder=os.path.expanduser("~")
global original_dir
original_dir = os.path.expanduser("~")
from tkinter import messagebox

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bsg_ide_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_xor_text_color(background_color):
    """
    Calculate optimal text color (black or white) for TikZ backgrounds using XOR rule.
    Uses perceived luminance and contrast ratio for better readability.

    Args:
        background_color: Tuple (R, G, B) or string color name

    Returns:
        'black' or 'white' based on background brightness
    """
    try:
        # Handle string color names by converting to RGB
        if isinstance(background_color, str):
            # Common color name mappings
            color_map = {
                'red': (255, 0, 0),
                'green': (0, 255, 0),
                'blue': (0, 0, 255),
                'yellow': (255, 255, 0),
                'cyan': (0, 255, 255),
                'magenta': (255, 0, 255),
                'white': (255, 255, 255),
                'black': (0, 0, 0),
                'gray': (128, 128, 128),
                'orange': (255, 165, 0),
                'purple': (128, 0, 128),
                'brown': (165, 42, 42),
                'pink': (255, 192, 203),
                'lightblue': (173, 216, 230),
                'lightgreen': (144, 238, 144),
                'lightgray': (211, 211, 211),
                'darkgray': (169, 169, 169),
                'darkblue': (0, 0, 139),
                'darkgreen': (0, 100, 0),
                'darkred': (139, 0, 0),
            }

            # Try to get from map, or parse hex/rgb string
            if background_color.lower() in color_map:
                r, g, b = color_map[background_color.lower()]
            elif background_color.startswith('#'):
                # Hex color
                hex_color = background_color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            elif background_color.startswith('rgb'):
                # rgb(r, g, b) format
                import re
                match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', background_color)
                if match:
                    r, g, b = map(int, match.groups())
                else:
                    # Default to medium gray if can't parse
                    r, g, b = 128, 128, 128
            else:
                # Unknown color, default to medium gray
                r, g, b = 128, 128, 128
        else:
            # Assume it's already an RGB tuple
            r, g, b = background_color

        # Ensure values are in 0-255 range
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))

        # Calculate perceived luminance using WCAG formula
        # Convert to linear RGB
        def to_linear(channel):
            c = channel / 255.0
            if c <= 0.04045:
                return c / 12.92
            else:
                return ((c + 0.055) / 1.055) ** 2.4

        r_linear = to_linear(r)
        g_linear = to_linear(g)
        b_linear = to_linear(b)

        # Calculate luminance
        luminance = 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear

        # XOR Rule: Use white text on dark backgrounds, black text on light backgrounds
        # Threshold can be adjusted (0.5 is middle gray)
        if luminance > 0.179:  # Adjusted threshold for better readability
            return 'black'
        else:
            return 'white'

    except Exception as e:
        print(f"Warning: Error calculating text color: {e}")
        # Fallback to black text
        return 'black'

def generate_default_requirements():
    """Generate a default requirements.txt if none found"""
    try:
        default_requirements = """# Core GUI dependencies
customtkinter==5.2.2
Pillow
tk

# Media and web dependencies
requests
yt_dlp
opencv-python
screeninfo
numpy

# PDF and document processing
PyMuPDF==1.23.7

# System utilities
pyautogui
pyspellchecker

# File type detection (not for Windows)
python-magic>=0.4.27; platform_system != "Windows"
"""
        # Try to write to user's local directory first
        save_paths = [
            Path.home() / '.local' / 'lib' / 'bsg-ide',
            Path.cwd(),
            Path(os.getenv('APPDATA', '')) / 'BSG-IDE'
        ]

        for path in save_paths:
            try:
                os.makedirs(path, exist_ok=True)
                req_file = path / 'requirements.txt'
                req_file.write_text(default_requirements)
                print(f"\nCreated default requirements.txt at: {req_file}")
                return req_file
            except:
                continue

        # If all save attempts fail, create in current directory
        print("\nWarning: Could not save to preferred locations")
        with open('requirements.txt', 'w') as f:
            f.write(default_requirements)
        return Path('requirements.txt')

    except Exception as e:
        print(f"Error generating requirements.txt: {str(e)}")
        return None
generate_default_requirements()

def launch_ide():
    """Entry point for command-line launcher"""
    try:
        # Ensure working directory is script directory
        import os
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Import required modules
        import sys
        from pathlib import Path

        # Add current directory to Python path
        current_dir = Path(__file__).parent.resolve()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))



        # Create and run the IDE
        app = BeamerSlideEditor()
        app.mainloop()

    except Exception as e:
        print(f"Error launching BSG-IDE: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def check_internet_connection():
    """Check internet connection without external dependencies"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def setup_virtual_env():
    """Setup and activate virtual environment with improved error handling"""
    try:
        home_dir = Path.home()
        venv_path = home_dir / 'my_python'

        # Create venv if it doesn't exist
        if not venv_path.exists():
            print(f"Creating virtual environment at {venv_path}")
            venv.create(venv_path, with_pip=True, clear=True)

        # Get paths based on platform
        if platform.system() == "Windows":
            venv_python = venv_path / "Scripts" / "python.exe"
            venv_pip = venv_path / "Scripts" / "pip.exe"
        else:
            venv_python = venv_path / "bin" / "python"
            venv_pip = venv_path / "bin" / "pip"

        # Verify the virtual environment is working
        if not venv_python.exists() or not venv_pip.exists():
            print("Virtual environment files not found, recreating...")
            shutil.rmtree(venv_path, ignore_errors=True)
            venv.create(venv_path, with_pip=True, clear=True)

        # Test virtual environment
        try:
            subprocess.run(
                [str(venv_python), "-c", "import sys; print(sys.prefix)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError:
            print("Virtual environment test failed, falling back to system Python")
            return sys.executable, "pip", False

        return str(venv_python), str(venv_pip), True

    except Exception as e:
        print(f"Error setting up virtual environment: {str(e)}")
        return sys.executable, "pip", False

def install_base_packages(pip_path):
    """Install essential packages without GUI dependencies"""
    base_packages = [
        "customtkinter",
        "Pillow",
        "tk",  # Basic tkinter
        "latexcodec",    # For LaTeX code handling
        "latex",         # Python LaTeX tools

    ]

    for package in base_packages:
        try:
            subprocess.run(
                [pip_path, "install", "--no-cache-dir", package],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except subprocess.CalledProcessError:
            continue


def get_requirements_path():
    """Get path to requirements.txt with exhaustive search"""
    possible_paths = [
        # Current directory
        Path.cwd() / 'requirements.txt',
        # Script directory
        Path(__file__).parent / 'requirements.txt',
        # User local installation
        Path.home() / '.local' / 'lib' / 'bsg-ide' / 'requirements.txt',
        # Windows AppData
        Path(os.getenv('APPDATA', '')) / 'BSG-IDE' / 'requirements.txt',
        # System-wide installation
        Path('/usr/local/share/bsg-ide/requirements.txt'),
        # Virtual environment
        Path(os.getenv('VIRTUAL_ENV', '')) / 'requirements.txt',
    ]

    # Add additional search paths for macOS
    if sys.platform == 'darwin':
        possible_paths.extend([
            Path.home() / 'Library' / 'Application Support' / 'BSG-IDE' / 'requirements.txt',
            Path('/Applications/BSG-IDE.app/Contents/Resources/requirements.txt')
        ])

    print("\nSearching for requirements.txt in:")
    for path in possible_paths:
        print(f"Checking {path}...")
        if path.exists():
            print(f"✓ Found requirements.txt at: {path}")
            return path

    print("Could not find requirements.txt in standard locations")
    return generate_default_requirements()


def install_system_dependencies():
    """Install system dependencies based on detected OS and package manager"""
    try:
        # Detect operating system
        if sys.platform.startswith('linux'):
            # Detect Linux distribution and package manager
            if shutil.which('apt'):  # Debian/Ubuntu
                mgr = 'apt'
                cmd = ['sudo', 'apt', 'install', '-y']
                deps = [
                    'python3-gi',
                    'python3-gi-cairo',
                    'gir1.2-gtk-3.0',
                    'python3-cairo',
                    'libgtk-3-0',
                    'librsvg2-common',
                    'poppler-utils',
                    'libgirepository1.0-dev',
                    'gcc',
                    'python3-dev',
                    'pkg-config',
                    'libcairo2-dev',
                    'pyautogui',
                    'gobject-introspection'
                ]
            elif shutil.which('dnf'):  # Fedora/RHEL
                mgr = 'dnf'
                cmd = ['sudo', 'dnf', 'install', '-y']
                deps = [
                    'python3-gobject',
                    'python3-cairo',
                    'gtk3',
                    'python3-devel',
                    'gcc',
                    'pkg-config',
                    'cairo-devel',
                    'pyautogui',
                    'gobject-introspection-devel',
                    'cairo-gobject-devel'
                ]
            elif shutil.which('pacman'):  # Arch Linux
                mgr = 'pacman'
                cmd = ['sudo', 'pacman', '-S', '--noconfirm']
                deps = [
                    'python-gobject',
                    'python-cairo',
                    'gtk3',
                    'python-pip',
                    'gcc',
                    'pkg-config',
                    'cairo',
                    'pyautogui',
                    'gobject-introspection'
                ]
            elif shutil.which('zypper'):  # openSUSE
                mgr = 'zypper'
                cmd = ['sudo', 'zypper', 'install', '-y']
                deps = [
                    'python3-gobject',
                    'python3-cairo',
                    'gtk3',
                    'python3-devel',
                    'gcc',
                    'pkg-config',
                    'cairo-devel',
                    'pyautogui',
                    'gobject-introspection-devel'
                ]
            else:
                print("Could not detect package manager. Please install dependencies manually:")
                print("Required: GTK3, Python-GObject, Cairo, and development tools")
                return False

            print(f"\nInstalling system dependencies using {mgr}...")
            try:
                subprocess.check_call(cmd + deps)
                print("✓ System dependencies installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                print(f"✗ Error installing system dependencies: {e}")
                return False

        elif sys.platform.startswith('darwin'):  # macOS
            try:
                # Check if Homebrew is installed
                if not shutil.which('brew'):
                    print("Homebrew not found. Installing...")
                    brew_install = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                    subprocess.check_call(brew_install, shell=True)

                # Install dependencies using Homebrew
                deps = [
                    'gtk+3',
                    'pygobject3',
                    'cairo',
                    'py3cairo',
                    'pyautogui',
                    'gobject-introspection'
                ]

                for dep in deps:
                    subprocess.check_call(['brew', 'install', dep])
                print("✓ System dependencies installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                print(f"✗ Error installing macOS dependencies: {e}")
                return False

        elif sys.platform.startswith('win'):  # Windows
            try:
                # For Windows, we'll use MSYS2 to install GTK and dependencies
                if not os.path.exists(r'C:\msys64'):
                    print("MSYS2 not found. Please install MSYS2 from: https://www.msys2.org/")
                    print("After installing, run: pacman -S mingw-w64-x86_64-gtk3 mingw-w64-x86_64-python-gobject")
                    return False

                # Update MSYS2 and install dependencies
                msys2_path = r'C:\msys64\usr\bin\bash.exe'
                deps = [
                    'mingw-w64-x86_64-gtk3',
                    'mingw-w64-x86_64-python-gobject',
                    'mingw-w64-x86_64-python-cairo',
                    'mingw-w64-x86_64-gcc'
                ]

                for dep in deps:
                    subprocess.check_call([msys2_path, '-lc', f'pacman -S --noconfirm {dep}'])
                print("✓ System dependencies installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                print(f"✗ Error installing Windows dependencies: {e}")
                return False

        return True

    except Exception as e:
        print(f"Error installing system dependencies: {str(e)}")
        return False
#------------------------------------------------------------
def install_bsg_ide(fix_mode=False):
    """Main installation function"""
    try:
        # Create a temporary root window if no parent is available
        temp_root = None
        try:
            # Try to get existing Tk root
            temp_root = tk.Tk()
            temp_root.withdraw()  # Hide the window
        except:
            pass

        installer = InstallationManager(temp_root)
        install_type = "system-wide" if installer.is_admin else "user"

        print(f"\nStarting BSG-IDE {'fix' if fix_mode else 'installation'} ({install_type})")
        print(f"Detected OS: {installer.system}")

        # Create directory structure
        installer.create_directory_structure()

        # Copy resources with enhanced PNG handling
        package_root = Path(__file__).resolve().parent
        required_files = {
            'BSG_IDE.py': ['base', 'python_site'],
            'BeamerSlideGenerator.py': ['base', 'python_site'],
            'requirements.txt': ['base'],
            'airis4d_logo.png': ['base/resources', 'resources', 'share', 'icons'],
            'bsg-ide.png': ['base/resources', 'resources', 'share', 'icons']
        }

        # Copy each file to its destinations
        for filename, destinations in required_files.items():
            source_file = package_root / filename
            if source_file.exists():
                for dest_type in destinations:
                    try:
                        # Handle nested paths
                        if '/' in dest_type:
                            base_type, subdir = dest_type.split('/')
                            if base_type in installer.install_paths:
                                dest_dir = installer.install_paths[base_type] / subdir
                        else:
                            dest_dir = installer.install_paths[dest_type]

                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = dest_dir / filename
                        shutil.copy2(source_file, dest_path)
                        print(f"✓ Copied {filename} to {dest_path}")
                    except Exception as e:
                        print(f"! Warning: Could not copy {filename} to {dest_type}: {e}")

        # Create launcher
        installer.create_launcher()

        # Setup OS integration
        installer.setup_os_integration()

        print(f"\nBSG-IDE {'fix' if fix_mode else 'installation'} completed successfully!")
        print("\nYou can now run BSG-IDE by:")
        if installer.system == "Windows":
            print("1. Using the Start Menu shortcut")
            print("2. Running 'bsg-ide' from Command Prompt")
        elif installer.system == "Darwin":
            print("1. Opening BSG-IDE from Applications")
            print("2. Running 'bsg-ide' from Terminal")
        else:
            print("1. Using the application menu")
            print("2. Running 'bsg-ide' from terminal")

        return True

    except Exception as e:
        print(f"\n✗ Error during {'fix' if fix_mode else 'installation'}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up temporary root window if we created one
        if temp_root is not None:
            try:
                temp_root.destroy()
            except:
                pass

def verify_installation():
    import site
    import sys
    from pathlib import Path

    """Verify BSG-IDE whether running from source or installed"""
    try:
        package_root, resources_dir = setup_paths()
        all_ok = True

        print("\nChecking installation...")
        print(f"Package root: {package_root}")
        print(f"Resources directory: {resources_dir}")

        # 1. Check core files
        required_files = {
            'BSG_IDE.py': package_root,
            'BeamerSlideGenerator.py': package_root,
            'requirements.txt': package_root,
            'airis4d_logo.png': resources_dir,
            'bsg-ide.png': resources_dir
        }

        print("\nChecking required files:")
        for filename, directory in required_files.items():
            path = directory / filename
            if path.exists():
                print(f"✓ Found {filename} in {directory}")
            else:
                print(f"✗ Missing {filename}")
                all_ok = False

        # 2. Check Python paths
        print("\nChecking Python paths:")
        if str(package_root) in sys.path:
            print("✓ Package root in Python path")
        else:
            print("✗ Package root not in Python path")
            all_ok = False

        try:

            # Get all possible site-packages locations
            site_packages_paths = {
                'user': Path(site.getusersitepackages()),
                'venv': Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages',
                'system': Path(site.getsitepackages()[0]) if site.getsitepackages() else None
            }

            found = False
            for location, path in site_packages_paths.items():
                if path and str(path) in sys.path:
                    print(f"✓ Found {location} site-packages in Python path: {path}")
                    found = True

            if not found:
                print("✗ No site-packages directories found in Python path")
                all_ok = False

            # Also check if BSG-IDE is importable
            try:
                import BeamerSlideGenerator
                print("✓ BSG-IDE package is importable")
            except ImportError as e:
                print(f"✗ Cannot import BSG-IDE package: {e}")
                all_ok = False

        except Exception as e:
            print(f"! Warning: Could not check site-packages: {e}")

        # 3. Check launcher script
        print("\nChecking launcher script:")
        if sys.platform == "win32":
            launcher_name = 'bsg-ide.bat'
        else:
            launcher_name = 'bsg-ide'

        launcher_locations = [
            Path.home() / '.local/bin',
            Path('/usr/local/bin'),
            package_root / 'bin'
        ]

        launcher_found = False
        for loc in launcher_locations:
            launcher_path = loc / launcher_name
            if launcher_path.exists():
                print(f"✓ Found launcher at: {launcher_path}")
                launcher_found = True
                # Check permissions on Unix-like systems
                if sys.platform != "win32":
                    if os.access(launcher_path, os.X_OK):
                        print("✓ Launcher has correct permissions")
                    else:
                        print("✗ Launcher missing executable permission")
                        all_ok = False
                break

        if not launcher_found:
            print("✗ Launcher script not found")
            all_ok = False

        # 4. Check desktop integration
        print("\nChecking desktop integration:")
        if sys.platform.startswith('linux'):
            # Check desktop entry
            desktop_locations = [
                Path.home() / '.local/share/applications',
                Path('/usr/share/applications')
            ]
            desktop_found = False
            for loc in desktop_locations:
                desktop_path = loc / 'bsg-ide.desktop'
                if desktop_path.exists():
                    print(f"✓ Found desktop entry at: {desktop_path}")
                    desktop_found = True
                    break

            if not desktop_found:
                print("✗ Desktop entry not found")
                all_ok = False

            # Check icons
            icon_found = False
            icon_sizes = ['16x16', '32x32', '48x48', '64x64', '128x128', '256x256']
            for size in icon_sizes:
                icon_locations = [
                    Path.home() / f'.local/share/icons/hicolor/{size}/apps',
                    Path(f'/usr/share/icons/hicolor/{size}/apps')
                ]
                for loc in icon_locations:
                    icon_path = loc / 'bsg-ide.png'
                    if icon_path.exists():
                        if not icon_found:  # Only print first found
                            print(f"✓ Found application icon at: {icon_path.parent.parent.parent}")
                        icon_found = True

            if not icon_found:
                print("✗ Application icon not found")
                all_ok = False

        elif sys.platform == "darwin":  # macOS
            app_locations = [
                Path.home() / 'Applications/BSG-IDE.app',
                Path('/Applications/BSG-IDE.app')
            ]
            app_found = False
            for loc in app_locations:
                if loc.exists():
                    print(f"✓ Found application bundle at: {loc}")
                    app_found = True
                    # Check Info.plist
                    if (loc / 'Contents/Info.plist').exists():
                        print("✓ Found Info.plist")
                    else:
                        print("✗ Missing Info.plist")
                        all_ok = False
                    break

            if not app_found:
                print("✗ Application bundle not found")
                all_ok = False

        elif sys.platform == "win32":  # Windows
            # Check Start Menu shortcut
            start_menu_locations = [
                Path(os.environ['APPDATA']) / 'Microsoft/Windows/Start Menu/Programs',
                Path(os.environ['PROGRAMDATA']) / 'Microsoft/Windows/Start Menu/Programs'
            ]
            shortcut_found = False
            for loc in start_menu_locations:
                shortcut_path = loc / 'BSG-IDE/BSG-IDE.lnk'
                if shortcut_path.exists():
                    print(f"✓ Found Start Menu shortcut at: {shortcut_path}")
                    shortcut_found = True
                    break

            if not shortcut_found:
                print("✗ Start Menu shortcut not found")
                all_ok = False

        # 5. Check imports
        print("\nChecking imports:")
        try:
            import BeamerSlideGenerator
            print("✓ Can import BeamerSlideGenerator")
        except ImportError as e:
            print(f"✗ Cannot import BeamerSlideGenerator: {e}")
            all_ok = False



        # Final status
        print("\nVerification result:")
        if all_ok:
            print("✓ All checks passed successfully!")
        else:
            print("✗ Some checks failed. Please run 'bsg-ide --fix' to repair the installation")

        return all_ok

    except Exception as e:
        print(f"\n✗ Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

"""
Path utilities for BSG-IDE to properly locate resources when installed via pip
"""

import os
import sys
from pathlib import Path
from importlib import resources

def get_package_root():
    """Get the root directory of the installed package"""
    try:
        # Try to get from module
        import bsg_ide
        return Path(bsg_ide.__file__).parent
    except ImportError:
        # Fallback to current file location
        return Path(__file__).parent.parent

def get_resource_path(filename):
    """Get path to a resource file in the package"""
    package_root = get_package_root()

    # Check various possible locations
    possible_paths = [
        package_root / filename,
        package_root / "resources" / filename,
        package_root.parent / filename,
        Path.cwd() / filename,
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    # If not found, return the package path as default
    return str(package_root / filename)

def ensure_bsg_file():
    """
    Ensure BeamerSlideGenerator.py is available by adding the package
    directory to Python path if needed
    """
    package_root = get_package_root()

    # Add package root to Python path if not already there
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    # Also add the parent directory
    parent_dir = package_root.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # Check if we can import BeamerSlideGenerator
    try:
        import BeamerSlideGenerator
        return True
    except ImportError:
        # Try direct import from package
        try:
            from bsg_ide import BeamerSlideGenerator
            return True
        except ImportError:
            return False


def get_package_root():
    """Get the package root directory whether installed or running from source"""
    try:
        # First check if we're running from source
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent

        # Check if we're in source directory (look for key files)
        if all((current_dir / f).exists() for f in ['BeamerSlideGenerator.py', 'BSG_IDE.py']):
            return current_dir

        # If not in source, check if we're pip installed
        try:
            import bsg_ide
            return Path(bsg_ide.__file__).parent
        except ImportError:
            pass

        # Finally check standard installation locations
        standard_locations = [
            Path.home() / '.local/lib/bsg-ide',
            Path.home() / '.local/share/bsg-ide',
            Path('/usr/local/share/bsg-ide')
        ]

        for loc in standard_locations:
            if (loc / 'BSG_IDE.py').exists():
                return loc

        # If we get here, return current directory as fallback
        return current_dir

    except Exception as e:
        print(f"Warning: Could not determine package root: {e}")
        return Path(__file__).parent

def setup_paths():
    """Setup Python paths for both source and installed runs"""
    package_root = get_package_root()

    # Add package root to Python path if not already there
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    # Find resources directory
    resources_dir = package_root / 'resources'
    if not resources_dir.exists():
        resources_dir = package_root  # Fallback to package root

    return package_root, resources_dir
#-----------------------------------------------------------


def create_desktop_entry():
    """Create desktop entry for pympress launcher"""
    if sys.platform.startswith('linux'):
        desktop_entry = """[Desktop Entry]
Type=Application
Name=pympress
Comment=PDF Presentation Tool
Exec=pympress %f
Icon=pympress
Terminal=false
Categories=Office;Presentation;
MimeType=application/pdf;
"""
        desktop_path = Path.home() / '.local' / 'share' / 'applications' / 'pympress.desktop'
        os.makedirs(desktop_path.parent, exist_ok=True)
        desktop_path.write_text(desktop_entry)
        desktop_path.chmod(0o755)
        print("✓ Desktop entry created")



def verify_existing_packages():
    """Verify existing packages when offline"""
    try:
        import customtkinter
        import PIL
        import tkinter
        print("✓ Required packages found in existing installation")
        return True
    except ImportError:
        print("Error: Required packages not found and cannot install offline")
        return False
#-----------------------------------------Check dependencies -------------------

def check_and_install_dependencies():
    """
    Two-phase dependency installation with improved virtual environment handling
    """
    # Suppress standard output during initial installation
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    try:
        # Check internet connection
        if not check_internet_connection():
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            print("No internet connection. Continuing with available packages.")
            return True

        # Get requirements.txt path
        requirements_path = get_requirements_path()
        if not requirements_path:
            print("Error: Could not locate or create requirements.txt")
            return False

        # Setup virtual environment
        #python_path, pip_path, venv_created = setup_virtual_env()

        # Phase 1: Install base packages (no GUI feedback)
        base_packages = {
            "customtkinter": "5.2.2",
            "Pillow": None,
            "tk": None
        }

        # First, upgrade pip in the virtual environment
        try:
            subprocess.run(
                [pip_path, "install", "--upgrade", "pip"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except Exception as e:
            print(f"Error upgrading pip: {e}")

        # Install base packages in virtual environment
        for package, version in base_packages.items():
            try:
                package_spec = f"{package}=={version}" if version else package
                # Do not use --user flag in virtual environment
                subprocess.run(
                    [pip_path, "install", "--no-cache-dir", package_spec],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                print(f"✓ Installed {package_spec}")
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")
                # Try alternative installation without version constraint
                try:
                    subprocess.run(
                        [pip_path, "install", "--no-cache-dir", package.split('==')[0]],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    print(f"✓ Installed {package} (latest version)")
                except subprocess.CalledProcessError as e2:
                    print(f"Alternative installation failed for {package}: {e2}")
                    continue

        # Create media_files directory
        os.makedirs('media_files', exist_ok=True)

        # Restore output for verification
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Verify customtkinter installation
        try:
            # Add virtual environment site-packages to Python path
            if venv_created:
                import site
                venv_site_packages = os.path.join(os.path.dirname(pip_path), '..', 'lib',
                                                f'python{sys.version_info.major}.{sys.version_info.minor}',
                                                'site-packages')
                sys.path.insert(0, venv_site_packages)

            import customtkinter
            print("✓ customtkinter installed successfully")

            # Phase 2: Install remaining packages with GUI feedback
            try:
                import tkinter as tk
                from tkinter import ttk
                install_remaining_packages(pip_path)
            except ImportError as e:
                print(f"Error importing GUI packages for phase 2: {e}")
                # Fall back to silent installation of remaining packages
                packages = {
                    'requests': 'requests',
                    'yt_dlp': 'yt-dlp',
                    'cv2': 'opencv-python',
                    'screeninfo': 'screeninfo',
                    'numpy': 'numpy',
                    'pyautogui':'pyautogui',
                    'fitz': 'PyMuPDF==1.23.7'

                }

                for import_name, install_name in packages.items():
                    try:
                        if not util.find_spec(import_name):
                            subprocess.run(
                                [pip_path, "install", "--no-cache-dir", install_name],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=True
                            )
                            print(f"✓ Installed {install_name}")
                    except:
                        continue

            return verify_installation()

        except ImportError:
            print("! Warning: customtkinter not found after installation")
            # Print the current sys.path for debugging
            print("\nCurrent Python path:")
            for p in sys.path:
                print(f"  {p}")
            return False

        return True

    except Exception as e:
        # Restore output
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Warning: Some dependencies may be missing. Continuing with available packages.")
        print(f"Error details: {str(e)}")
        return True

    finally:
        # Ensure output is restored
        sys.stdout = original_stdout
        sys.stderr = original_stderr

def install_remaining_packages(pip_path):
    """Install all remaining packages from requirements.txt with progress feedback"""
    try:
        # Now we can safely import GUI packages
        import tkinter as tk
        from tkinter import ttk
        import customtkinter as ctk

        class ProgressDialog:
            def __init__(self):
                self.root = ctk.CTk()
                self.root.title("Installing Dependencies")
                self.root.geometry("300x150")

                # Center window
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                x = (screen_width - 300) // 2
                y = (screen_height - 150) // 2
                self.root.geometry(f"+{x}+{y}")

                self.label = ctk.CTkLabel(
                    self.root,
                    text="Installing dependencies...",
                    font=("Arial", 12)
                )
                self.label.pack(pady=20)

                self.progress = ctk.CTkProgressBar(self.root)
                self.progress.pack(pady=10, padx=20, fill="x")
                self.progress.set(0)

                self.status = ctk.CTkLabel(
                    self.root,
                    text="",
                    font=("Arial", 10)
                )
                self.status.pack(pady=5)

                self.root.update()

            def update(self, progress, text=None):
                self.progress.set(progress)
                if text:
                    self.label.configure(text=text)
                    # Also update status with package name
                    if "Installing" in text:
                        package = text.split("Installing ")[-1].strip('...')
                        self.status.configure(text=f"Package: {package}")
                self.root.update()

            def close(self):
                self.root.destroy()

        try:
            # Get requirements.txt path
            requirements_path = get_requirements_path()

            # Read requirements.txt
            with open(requirements_path, 'r') as f:
                requirements = [line.strip() for line in f
                              if line.strip() and not line.startswith('#')]

            # Create progress dialog
            dialog = ProgressDialog()

            # Install packages with progress updates
            total = len(requirements)
            successful = 0
            failed = []

            for i, requirement in enumerate(requirements, 1):
                progress = i / total
                package_name = requirement.split('>=')[0].split('==')[0].strip()
                dialog.update(progress, f"Installing {package_name}...")

                try:
                    # First try to import to check if already installed
                    try:
                        if ';' in requirement:  # Skip platform-specific requirements check
                            raise ImportError
                        module_name = package_name.replace('-', '_')
                        __import__(module_name)
                        successful += 1
                        dialog.status.configure(text=f"✓ {package_name} already installed")
                        dialog.root.update()
                        continue
                    except ImportError:
                        pass

                    # If not installed, install it
                    result = subprocess.run(
                        [pip_path, "install", "--no-cache-dir", requirement],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
                    successful += 1
                    dialog.status.configure(text=f"✓ {package_name} installed successfully")
                except Exception as e:
                    print(f"Warning: Failed to install {requirement}: {str(e)}")
                    failed.append(package_name)
                    dialog.status.configure(text=f"✗ Failed to install {package_name}")

                dialog.root.update()

            # Show completion status
            dialog.update(1.0, "Installation complete!")
            final_status = f"Installed {successful} of {total} packages"
            if failed:
                final_status += f"\nFailed: {', '.join(failed)}"
            dialog.status.configure(text=final_status)
            dialog.root.after(2000, dialog.close)  # Close after 2 seconds
            dialog.root.mainloop()

        except Exception as e:
            if 'dialog' in locals():
                dialog.close()
            print(f"Error in GUI installation: {str(e)}")
            # Fall back to silent installation
            subprocess.run(
                [pip_path, "install", "-r", str(requirements_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

    except ImportError as e:
        print(f"GUI packages not available for progress dialog: {str(e)}")
        print("Falling back to silent installation...")
        # Fall back to silent installation
        subprocess.run(
            [pip_path, "install", "-r", str(requirements_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        print(f"Error installing packages: {str(e)}")



#check_and_install_dependencies()


import atexit
import shutil
import threading
import queue
import socket
import json
import time
import zipfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import customtkinter as ctk
from typing import Optional, Dict, List, Tuple,Any
from pathlib import Path
from PIL import Image
import traceback
import webbrowser
#from BSE import BeamerSlideEditor,BeamerSyntaxHighlighter

#-------------------------------------BSE-----------------------------------------------

import customtkinter as ctk
import tkinter as tk
import os,sys
import threading
from pathlib import Path
from PIL import Image
#from BSE_ITR import InteractiveTerminal
ENHANCED_FEATURES_AVAILABLE = False
try:
    from .EnhancedCommandDialog import EnhancedCommandIndexDialog, IntelligentAutocomplete, LatexCommandHelper, CommandTooltip
    ENHANCED_FEATURES_AVAILABLE = True
    from .Grammarly import  GrammarlyIntegration,GrammarlySetupDialog,AutomatedGrammarlyIntegration
    from .InteractiveTerminal import InteractiveTerminal
    print("✓ Enhanced command features loaded")
except ImportError as e:
    print(f"Enhanced features not available: {e}")
    from EnhancedCommandDialog import EnhancedCommandIndexDialog, IntelligentAutocomplete, LatexCommandHelper, CommandTooltip
    ENHANCED_FEATURES_AVAILABLE = True
    from Grammarly import  GrammarlyIntegration,GrammarlySetupDialog,AutomatedGrammarlyIntegration
    from InteractiveTerminal import InteractiveTerminal



#---------------Helper Utils ------------------------------
class SessionManager:
    """Manages persistence of session data between IDE launches"""

    def __init__(self):
        try:
            # Get user's home directory
            self.home_dir = Path.home()
            # Create .bsg-ide directory in user's home if it doesn't exist
            self.config_dir = self.home_dir / '.bsg-ide'
            self.config_dir.mkdir(exist_ok=True)
            self.session_file = self.config_dir / 'session.json'

            # Try to find Documents folder
            documents_dir = None
            possible_docs = [
                self.home_dir / 'Documents',  # Linux/macOS
                self.home_dir / 'documents',  # Alternative spelling
                Path(os.path.expandvars('%USERPROFILE%\\Documents'))  # Windows
            ]

            for doc_path in possible_docs:
                if doc_path.exists() and doc_path.is_dir():
                    documents_dir = doc_path
                    break

            # Fall back to home directory if Documents not found
            default_dir = str(documents_dir) if documents_dir else str(self.home_dir)

            self.default_session = {
                'last_file': None,
                'working_directory': default_dir,
                'recent_files': [],
                'window_size': {'width': 1200, 'height': 800},
                'window_position': {'x': None, 'y': None}
            }
        except Exception as e:
            print(f"Warning: Could not initialize session manager: {str(e)}")
            # Still allow the program to run with defaults
            self.session_file = None
            self.default_session = {
                'last_file': None,
                'working_directory': str(Path.home()),
                'recent_files': [],
                'window_size': {'width': 1200, 'height': 800},
                'window_position': {'x': None, 'y': None}
            }

    def load_session(self):
        """Load session data from file with Documents folder preference"""
        try:
            if self.session_file and self.session_file.exists():
                with open(self.session_file, 'r') as f:
                    data = json.load(f)

                # Validate loaded data
                session_data = self.default_session.copy()

                # Only update working directory if it exists and there's a last file or recent files
                if data.get('last_file') or data.get('recent_files'):
                    if os.path.exists(data.get('working_directory', '')):
                        session_data.update({k: v for k, v in data.items() if k in self.default_session})

                # Filter out non-existent recent files
                session_data['recent_files'] = [
                    f for f in session_data['recent_files']
                    if os.path.exists(f)
                ]

                return session_data

            return self.default_session.copy()

        except Exception as e:
            print(f"Warning: Could not load session data: {str(e)}")
            return self.default_session.copy()

    def save_session(self, data):
        """Save session data to file"""
        if not self.session_file:
            return  # Skip saving if no session file available

        try:
            # Ensure all paths are strings
            session_data = {
                'last_file': str(data.get('last_file')) if data.get('last_file') else None,
                'working_directory': str(data.get('working_directory', self.default_session['working_directory'])),
                'recent_files': [str(f) for f in data.get('recent_files', [])[-10:]],  # Keep last 10 files
                'window_size': data.get('window_size', self.default_session['window_size']),
                'window_position': data.get('window_position', self.default_session['window_position'])
            }

            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save session data: {str(e)}")


class MediaURLDialog(ctk.CTkToplevel):
    def __init__(self, parent, slide_index, media_entry):
        super().__init__(parent)
        self.title("Update Media Location")
        self.geometry("500x150")
        self.media_entry = media_entry

        # Center dialog
        self.transient(parent)
        self.grab_set()

        # Create widgets
        ctk.CTkLabel(self, text=f"Enter media URL for slide {slide_index + 1}:").pack(pady=10)

        self.url_entry = ctk.CTkEntry(self, width=400)
        self.url_entry.pack(pady=10)
        self.url_entry.insert(0, media_entry.get())

        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=10)

        ctk.CTkButton(button_frame, text="Play URL",
                     command=self.use_play_url).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Static URL",
                     command=self.use_static_url).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Cancel",
                     command=self.cancel).pack(side="left", padx=5)

    def use_play_url(self):
        url = self.url_entry.get().strip()
        if url:
            self.media_entry.delete(0, 'end')
            self.media_entry.insert(0, f"\\play \\url {url}")
        self.destroy()

    def use_static_url(self):
        url = self.url_entry.get().strip()
        if url:
            self.media_entry.delete(0, 'end')
            self.media_entry.insert(0, f"\\url {url}")
        self.destroy()

    def cancel(self):
        self.destroy()

#--------------------------------------------------Dialogs -------------------------
class InstitutionNameDialog(ctk.CTkToplevel):
    """Dialog for handling long institution names"""
    def __init__(self, parent, institution_name):
        super().__init__(parent)
        self.title("Institution Name Warning")
        self.geometry("500x250")
        self.short_name = None

        # Center dialog
        self.transient(parent)
        self.grab_set()

        # Create widgets
        ctk.CTkLabel(self, text="Long Institution Name Detected",
                    font=("Arial", 14, "bold")).pack(pady=10)

        ctk.CTkLabel(self, text=f"Current name:\n{institution_name}",
                    wraplength=450).pack(pady=10)

        ctk.CTkLabel(self, text="Please provide a shorter version for slide footers:").pack(pady=5)

        self.entry = ctk.CTkEntry(self, width=300)
        self.entry.pack(pady=10)

        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=20)

        ctk.CTkButton(button_frame, text="Use Short Name",
                     command=self.use_short_name).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Keep Original",
                     command=self.keep_original).pack(side="left", padx=10)

    def use_short_name(self):
        self.short_name = self.entry.get()
        self.destroy()

    def keep_original(self):
        self.destroy()

class MediaSelectionDialog(ctk.CTkToplevel):
    """Dialog for selecting media when URL fails"""
    def __init__(self, parent, title, content):
        super().__init__(parent)
        self.title("Media Selection")
        self.geometry("600x400")
        self.result = None

        # Center dialog
        self.transient(parent)
        self.grab_set()

        # Create widgets
        ctk.CTkLabel(self, text="Media Selection Required",
                    font=("Arial", 14, "bold")).pack(pady=10)

        # Show search query
        search_query = construct_search_query(title, content)
        query_frame = ctk.CTkFrame(self)
        query_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(query_frame, text=f"Search query: {search_query}",
                    wraplength=550).pack(side="left", pady=5)

        ctk.CTkButton(query_frame, text="Open Search",
                     command=lambda: open_google_image_search(search_query)).pack(side="right", padx=5)

        # Options
        options_frame = ctk.CTkFrame(self)
        options_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # URL Entry
        url_frame = ctk.CTkFrame(options_frame)
        url_frame.pack(fill="x", pady=5)
        self.url_entry = ctk.CTkEntry(url_frame, width=400)
        self.url_entry.pack(side="left", padx=5)
        ctk.CTkButton(url_frame, text="Use URL",
                     command=self.use_url).pack(side="left", padx=5)

        # File Selection
        file_frame = ctk.CTkFrame(options_frame)
        file_frame.pack(fill="x", pady=5)
        self.file_listbox = ctk.CTkTextbox(file_frame, height=150)
        self.file_listbox.pack(fill="x", pady=5)

        # Populate file list
        try:
            files = os.listdir('media_files')
            for i, file in enumerate(files, 1):
                self.file_listbox.insert('end', f"{i}. {file}\n")
        except Exception as e:
            self.file_listbox.insert('end', f"Error accessing media_files: {str(e)}")

        ctk.CTkButton(file_frame, text="Use Selected File",
                     command=self.use_file).pack(pady=5)

        # No Media Option
        ctk.CTkButton(options_frame, text="Create Slide Without Media",
                     command=self.use_none).pack(pady=10)

    def use_url(self):
        url = self.url_entry.get().strip()
        if url:
            self.result = url
            self.destroy()

    def use_file(self):
        # Get selected line
        try:
            selection = self.file_listbox.get("sel.first", "sel.last")
            if selection:
                file_name = selection.split('.', 1)[1].strip()
                self.result = f"\\file media_files/{file_name}"
                self.destroy()
        except:
            messagebox.showwarning("Selection Required",
                                 "Please select a file from the list")

    def use_none(self):
        self.result = "\\None"
        self.destroy()

#------------------------------------------------------------------------------------------
class NotesToolbar(ctk.CTkFrame):
    """Toolbar for notes formatting and templates"""
    def __init__(self, parent, notes_editor, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.notes_editor = notes_editor

        # Templates
        self.templates = {
            "Key Points": "• Key points:\n  - \n  - \n  - \n",
            "Time Markers": "• Timing guide:\n  0:00 - Introduction\n  0:00 - Main points\n  0:00 - Conclusion",
            "Questions": "• Potential questions:\nQ1: \nA1: \n\nQ2: \nA2: ",
            "References": "• Additional references:\n  - Title:\n    Author:\n    Page: ",
            "Technical Details": "• Technical details:\n  - Specifications:\n  - Parameters:\n  - Requirements:",
        }

        self.create_toolbar()

    def create_toolbar(self):
        """Create the notes toolbar"""
        # Template dropdown
        template_frame = ctk.CTkFrame(self)
        template_frame.pack(side="left", padx=5, pady=2)

        ctk.CTkLabel(template_frame, text="Template:").pack(side="left", padx=2)

        self.template_var = tk.StringVar(value="Select Template")
        template_menu = ctk.CTkOptionMenu(
            template_frame,
            values=list(self.templates.keys()),
            variable=self.template_var,
            command=self.insert_template,
            width=150
        )
        template_menu.pack(side="left", padx=2)

        # Separator
        ttk.Separator(self, orient="vertical").pack(side="left", padx=5, fill="y", pady=2)

        # Formatting buttons
        formatting_frame = ctk.CTkFrame(self)
        formatting_frame.pack(side="left", padx=5, pady=2)

        formatting_buttons = [
            ("B", self.add_bold, "Bold"),
            ("I", self.add_italic, "Italic"),
            ("C", self.add_color, "Color"),
            ("⚡", self.add_highlight, "Highlight"),
            ("•", self.add_bullet, "Bullet point"),
            ("⏱", self.add_timestamp, "Timestamp"),
            ("⚠", self.add_alert, "Alert"),
            ("💡", self.add_tip, "Tip")
        ]

        for text, command, tooltip in formatting_buttons:
            btn = ctk.CTkButton(
                formatting_frame,
                text=text,
                command=command,
                width=30,
                height=30
            )
            btn.pack(side="left", padx=2)
            self.create_tooltip(btn, tooltip)

    def create_tooltip(self, widget, text):
        """Create tooltip for buttons"""
        def show_tooltip(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20

            # Create tooltip window
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")

            label = tk.Label(self.tooltip, text=text,
                           justify='left',
                           background="#ffffe0", relief='solid', borderwidth=1)
            label.pack()

        def hide_tooltip(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()

        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)

    def insert_template(self, choice):
        """Insert selected template"""
        if choice in self.templates:
            self.notes_editor.insert('insert', self.templates[choice])
            self.template_var.set("Select Template")  # Reset dropdown

    def add_bold(self):
        """Add bold text"""
        self.wrap_selection(r'\textbf{', '}')

    def add_italic(self):
        """Add italic text"""
        self.wrap_selection(r'\textit{', '}')

    def add_color(self):
        """Add colored text"""
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        color = simpledialog.askstring(
            "Color",
            "Enter color name or RGB values:",
            initialvalue=colors[0]
        )
        if color:
            self.wrap_selection(f'\\textcolor{{{color}}}{{', '}')

    def add_highlight(self):
        """Add highlighted text"""
        self.wrap_selection('\\hl{', '}')

    def add_bullet(self):
        """Add bullet point"""
        self.notes_editor.insert('insert', '\n• ')

    def add_timestamp(self):
        """Add timestamp"""
        timestamp = simpledialog.askstring(
            "Timestamp",
            "Enter timestamp (MM:SS):",
            initialvalue="00:00"
        )
        if timestamp:
            self.notes_editor.insert('insert', f'[{timestamp}] ')

    def add_alert(self):
        """Add alert note"""
        self.notes_editor.insert('insert', '⚠ Important: ')

    def add_tip(self):
        """Add tip"""
        self.notes_editor.insert('insert', '💡 Tip: ')

    def wrap_selection(self, prefix, suffix):
        """Wrap selected text with prefix and suffix"""
        try:
            selection = self.notes_editor.get('sel.first', 'sel.last')
            self.notes_editor.delete('sel.first', 'sel.last')
            self.notes_editor.insert('insert', f'{prefix}{selection}{suffix}')
        except tk.TclError:  # No selection
            self.notes_editor.insert('insert', f'{prefix}{suffix}')
            # Move cursor inside braces
            current_pos = self.notes_editor.index('insert')
            self.notes_editor.mark_set('insert', f'{current_pos}-{len(suffix)}c')

class EnhancedNotesEditor(ctk.CTkFrame):
    """Enhanced notes editor with toolbar and templates"""
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Create toolbar
        self.toolbar = NotesToolbar(self, self.notes_editor)
        self.toolbar.pack(fill="x", padx=2, pady=2)

        # Create editor
        self.notes_editor = ctk.CTkTextbox(self)
        self.notes_editor.pack(fill="both", expand=True, padx=2, pady=2)

        # Enhanced syntax highlighting
        self.setup_syntax_highlighting()

    def setup_syntax_highlighting(self):
        """Setup enhanced syntax highlighting for notes"""
        self.highlighter = BeamerSyntaxHighlighter(self.notes_editor)

        # Add additional patterns for notes
        additional_patterns = [
            (r'⚠.*$', 'alert'),
            (r'💡.*$', 'tip'),
            (r'\[[\d:]+\]', 'timestamp'),
            (r'•.*$', 'bullet'),
            (r'\\hl\{.*?\}', 'highlight'),
        ]

        # Add additional colors
        additional_colors = {
            'alert': '#FF6B6B',
            'tip': '#4ECDC4',
            'timestamp': '#FFB86C',
            'highlight': '#BD93F9',
        }

        # Update highlighter
        self.highlighter.patterns.extend(additional_patterns)
        self.highlighter.colors.update(additional_colors)

#------------------------------------------------------------------------------------------
class FileThumbnailBrowser(ctk.CTkToplevel):
    def __init__(self, parent, initial_dir="media_files", callback=None):
        super().__init__(parent)

        # Import required modules
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.Image = Image
            self.ImageDraw = ImageDraw
            self.ImageFont = ImageFont
            self.has_pil = True
        except ImportError as e:
            print(f"Error importing PIL modules: {e}")
            self.has_pil = False
            messagebox.showwarning("Warning",
                                 "Image processing libraries not available.\nThumbnails will be limited.")

        self.title("Media Browser")
        self.geometry("800x600")

        # Store initial directory and callback
        self.current_dir = os.path.abspath(initial_dir)
        self.callback = callback
        self.thumbnails = []
        self.current_row = 0
        self.current_col = 0
        self.max_cols = 4

        # Create media_files directory if it doesn't exist
        os.makedirs(initial_dir, exist_ok=True)

        # File categories with extended video types
        self.file_categories = {
            'image': ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'),
            'video': ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.gif'),
            'audio': ('.mp3', '.wav', '.ogg', '.m4a', '.flac'),
            'document': ('.pdf', '.doc', '.docx', '.txt', '.tex'),
            'data': ('.csv', '.xlsx', '.json', '.xml')
        }

        # Create UI components
        self.create_navigation_bar()
        self.create_toolbar()
        self.create_content_area()
        self.load_files()
        # Add destruction handler
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Destroy>", self._on_destroy)

    def _on_close(self):
        """Handle window closing"""
        self._cleanup_bindings()
        self.destroy()

    def _on_destroy(self, event):
        """Handle widget destruction"""
        if event.widget == self:
            self._cleanup_bindings()

    def _cleanup_bindings(self):
        """Clean up all global bindings"""
        try:
            # Unbind all global mouse wheel events
            self.unbind_all("<MouseWheel>")
            if sys.platform.startswith('linux'):
                self.unbind_all("<Button-4>")
                self.unbind_all("<Button-5>")
        except:
            pass  # Ignore errors during cleanup

    def _bind_mousewheel(self, event):
        """Bind mousewheel when mouse enters canvas - with safety check"""
        if not self.winfo_exists():  # Check if widget still exists
            return

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        if sys.platform.startswith('linux'):
            self.canvas.bind_all("<Button-4>", self._on_mousewheel)
            self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        """Handle mouse wheel and touchpad scrolling with safety check"""
        if not self.winfo_exists() or not self.canvas.winfo_exists():
            return  # Don't process if widgets are destroyed


    def create_thumbnail(self, file_path):
        """Create thumbnail with proper error handling"""
        if not self.has_pil:
            return self.create_fallback_thumbnail()

        try:
            category = self.get_file_category(file_path)
            thumb_size = (150, 150)

            if category == 'image':
                try:
                    with self.Image.open(file_path) as img:
                        # Convert to RGB if necessary
                        if img.mode in ('RGBA', 'P'):
                            img = img.convert('RGB')

                        # Create thumbnail
                        img.thumbnail(thumb_size, self.Image.Resampling.LANCZOS)

                        # Create background
                        thumb_bg = self.Image.new('RGB', thumb_size, 'black')

                        # Center image on background
                        offset = ((thumb_size[0] - img.size[0]) // 2,
                                (thumb_size[1] - img.size[1]) // 2)
                        thumb_bg.paste(img, offset)

                        return ctk.CTkImage(light_image=thumb_bg,
                                          dark_image=thumb_bg,
                                          size=thumb_size)
                except Exception as e:
                    print(f"Error creating image thumbnail: {str(e)}")
                    return self.create_generic_thumbnail("Image\nError", "#8B0000")

            else:
                # Create appropriate generic thumbnail based on category
                colors = {
                    'video': "#4a90e2",
                    'audio': "#e24a90",
                    'document': "#90e24a",
                    'data': "#4ae290"
                }
                color = colors.get(category, "#808080")
                text = category.upper() if category else "FILE"
                return self.create_generic_thumbnail(text, color)

        except Exception as e:
            print(f"Error creating thumbnail for {file_path}: {str(e)}")
            return self.create_fallback_thumbnail()

    def create_generic_thumbnail(self, text, color):
        """Create generic thumbnail with text"""
        if not self.has_pil:
            return self.create_fallback_thumbnail()

        try:
            thumb_size = (150, 150)
            img = self.Image.new('RGB', thumb_size, 'black')
            draw = self.ImageDraw.Draw(img)

            # Draw colored rectangle
            margin = 20
            draw.rectangle(
                [margin, margin, thumb_size[0]-margin, thumb_size[1]-margin],
                fill=color
            )

            # Draw text
            text_bbox = draw.textbbox((0, 0), text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = (thumb_size[0] - text_width) // 2
            text_y = (thumb_size[1] - text_height) // 2

            draw.text((text_x, text_y), text, fill="white")

            return ctk.CTkImage(light_image=img,
                              dark_image=img,
                              size=thumb_size)
        except Exception as e:
            print(f"Error creating generic thumbnail: {str(e)}")
            return self.create_fallback_thumbnail()

    def create_fallback_thumbnail(self):
        """Create a basic fallback thumbnail when PIL is not available or errors occur"""
        try:
            img = self.Image.new('RGB', (150, 150), color='gray')
            return ctk.CTkImage(light_image=img,
                              dark_image=img,
                              size=(150, 150))
        except:
            # Create an empty CTkImage if all else fails
            return ctk.CTkImage(light_image=None,
                              dark_image=None,
                              size=(150, 150))

#-------------------------------------------------------------------------------------------
    def create_file_item(self, file_name):
        """Create file display item with proper error handling"""
        try:
            frame = ctk.CTkFrame(self.scrollable_frame)
            frame.grid(row=self.current_row, column=self.current_col,
                      padx=10, pady=10, sticky="nsew")

            file_path = os.path.join(self.current_dir, file_name)

            # Create thumbnail
            try:
                thumbnail = self.create_thumbnail(file_path)
            except Exception as e:
                print(f"Error creating thumbnail: {e}")
                thumbnail = self.create_generic_thumbnail("Error", "#8B0000")

            if thumbnail:
                # Create thumbnail button
                thumb_button = ctk.CTkButton(
                    frame,
                    image=thumbnail,
                    text="",
                    command=lambda path=file_path: self.on_file_click(path),
                    width=150,
                    height=150
                )
                thumb_button.pack(pady=(5, 0))

                # Add filename label
                label = ctk.CTkLabel(
                    frame,
                    text=file_name,
                    wraplength=140
                )
                label.pack(pady=(5, 5))

                # Store reference to thumbnail
                self.thumbnails.append(thumbnail)

            # Update grid position
            self.current_col += 1
            if self.current_col >= self.max_cols:
                self.current_col = 0
                self.current_row += 1

        except Exception as e:
            print(f"Error creating file item: {str(e)}")

    def on_file_click(self, file_path: str) -> None:
        """Handle file selection with proper path handling"""
        if self.callback:
            # Create relative path if file is in media_files directory
            try:
                relative_to_media = os.path.relpath(file_path, 'media_files')
                if relative_to_media.startswith('..'):
                    # File is outside media_files - use absolute path
                    final_path = file_path
                else:
                    # File is inside media_files - use relative path
                    final_path = os.path.join('media_files', relative_to_media)

                # Determine if file should be played
                ext = os.path.splitext(file_path)[1].lower()
                is_video = ext in self.file_categories['video']

                if is_video and hasattr(self, 'play_vars') and self.play_vars.get(file_path, tk.BooleanVar(value=True)).get():
                    self.callback(f"\\play \\file {final_path}")
                else:
                    self.callback(f"\\file {final_path}")

            except Exception as e:
                print(f"Error handling file selection: {str(e)}")
                return

        self.destroy()

    def create_navigation_bar(self):
        """Create navigation bar with path and controls"""
        nav_frame = ctk.CTkFrame(self)
        nav_frame.pack(fill="x", padx=5, pady=5)

        # Back button
        self.back_button = ctk.CTkButton(
            nav_frame,
            text="⬅ Back",
            command=self.navigate_up,
            width=60
        )
        self.back_button.pack(side="left", padx=5)

        # Path display and navigation
        self.path_var = tk.StringVar()
        self.path_entry = ctk.CTkEntry(
            nav_frame,
            textvariable=self.path_var,
            width=400
        )
        self.path_entry.pack(side="left", fill="x", expand=True, padx=5)
        self.path_entry.bind('<Return>', self.navigate_to_path)

        # Update current path
        self.update_path_display()

    def create_toolbar(self):
        """Create toolbar with sorting and view options"""
        toolbar = ctk.CTkFrame(self)
        toolbar.pack(fill="x", padx=5, pady=5)

        # Sorting options
        sort_label = ctk.CTkLabel(toolbar, text="Sort by:")
        sort_label.pack(side="left", padx=5)

        self.sort_var = tk.StringVar(value="name")
        sort_options = ["name", "date", "size", "type"]

        for option in sort_options:
            rb = ctk.CTkRadioButton(
                toolbar,
                text=option.capitalize(),
                variable=self.sort_var,
                value=option,
                command=self.refresh_files
            )
            rb.pack(side="left", padx=10)

        # Sort direction
        self.reverse_var = tk.BooleanVar(value=False)
        reverse_cb = ctk.CTkCheckBox(
            toolbar,
            text="Reverse",
            variable=self.reverse_var,
            command=self.refresh_files
        )
        reverse_cb.pack(side="left", padx=10)

    def create_content_area(self):
        """Create scrollable content area with enhanced navigation"""
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create canvas with scrollbars
        self.canvas = tk.Canvas(self.main_frame, bg='black')
        self.v_scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical")
        self.h_scrollbar = ttk.Scrollbar(self.main_frame, orient="horizontal")

        # Configure scrollbars
        self.v_scrollbar.config(command=self.canvas.yview)
        self.h_scrollbar.config(command=self.canvas.xview)
        self.canvas.config(
            yscrollcommand=self.v_scrollbar.set,
            xscrollcommand=self.h_scrollbar.set
        )

        # Pack scrollbars
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Create frame for content
        self.scrollable_frame = ctk.CTkFrame(self.canvas)
        self.canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw",
            tags="self.scrollable_frame"
        )

        # Configure scroll bindings
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Bind scroll events
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        # Touch pad/track pad scrolling
        if sys.platform == 'darwin':
            self.canvas.bind("<TouchpadScroll>", self._on_touchpad_scroll)
        else:
            self.canvas.bind("<Shift-MouseWheel>", self._on_touchpad_scroll)

    def _on_mousewheel(self, event):
        """Handle mouse wheel and touchpad scrolling"""
        if event.num == 4:  # Linux up
            delta = 120
        elif event.num == 5:  # Linux down
            delta = -120
        else:  # Windows/MacOS
            delta = event.delta

        shift_pressed = event.state & 0x1  # Check if Shift is pressed
        if shift_pressed:
            self.canvas.xview_scroll(int(-1 * delta/120), "units")
        else:
            self.canvas.yview_scroll(int(-1 * delta/120), "units")

    def _on_touchpad_scroll(self, event):
        """Handle touchpad scrolling"""
        if event.state & 0x1:  # Shift pressed - horizontal scroll
            self.canvas.xview_scroll(int(-1 * event.delta/30), "units")
        else:  # Vertical scroll
            self.canvas.yview_scroll(int(-1 * event.delta/30), "units")

    def _bind_mousewheel(self, event):
        """Bind mousewheel when mouse enters canvas"""
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        if sys.platform.startswith('linux'):
            self.canvas.bind_all("<Button-4>", self._on_mousewheel)
            self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        """Unbind mousewheel when mouse leaves canvas"""
        self.canvas.unbind_all("<MouseWheel>")
        if sys.platform.startswith('linux'):
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

    def get_file_category(self, filename):
        """Determine file category and appropriate thumbnail style"""
        ext = os.path.splitext(filename)[1].lower()

        for category, extensions in self.file_categories.items():
            if ext in extensions:
                return category

        return 'other'


    def navigate_up(self):
        """Navigate to parent directory"""
        parent = os.path.dirname(self.current_dir)
        if os.path.exists(parent):
            self.current_dir = parent
            self.update_path_display()
            self.load_files()

    def navigate_to_path(self, event=None):
        """Navigate to entered path"""
        new_path = self.path_var.get()
        if os.path.exists(new_path):
            self.current_dir = os.path.abspath(new_path)
            self.update_path_display()
            self.load_files()
        else:
            messagebox.showerror("Error", "Invalid path")
            self.update_path_display()

    def update_path_display(self):
        """Update path display"""
        self.path_var.set(self.current_dir)

    def load_files(self):
        """Load files and folders with enhanced display"""
        # Clear existing display
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnails.clear()
        self.current_row = 0
        self.current_col = 0

        try:
            # Get directories and files
            entries = os.listdir(self.current_dir)
            folders = []
            files = []

            for entry in entries:
                full_path = os.path.join(self.current_dir, entry)
                if os.path.isdir(full_path):
                    folders.append(entry)
                else:
                    files.append(entry)

            # Sort folders and files separately
            folders.sort()
            files = self.sort_files(files)

            # Display folders first
            for folder in folders:
                self.create_folder_item(folder)

            # Then display files
            for file in files:
                self.create_file_item(file)

        except Exception as e:
            messagebox.showerror("Error", f"Error loading directory: {str(e)}")

    def create_folder_item(self, folder_name):
        """Create folder display item"""
        frame = ctk.CTkFrame(self.scrollable_frame)
        frame.grid(row=self.current_row, column=self.current_col,
                  padx=10, pady=10, sticky="nsew")

        # Create folder button with icon
        folder_button = ctk.CTkButton(
            frame,
            text="📁",
            command=lambda f=folder_name: self.enter_folder(f),
            width=150,
            height=150
        )
        folder_button.pack(pady=(5, 0))

        # Add folder name label
        label = ctk.CTkLabel(
            frame,
            text=folder_name,
            wraplength=140
        )
        label.pack(pady=(5, 5))

        # Update grid position
        self.current_col += 1
        if self.current_col >= self.max_cols:
            self.current_col = 0
            self.current_row += 1


    def enter_folder(self, folder_name):
        """Enter selected folder"""
        new_path = os.path.join(self.current_dir, folder_name)
        if os.path.exists(new_path):
            self.current_dir = new_path
            self.update_path_display()
            self.load_files()

    def sort_files(self, files):
        """Sort files based on current criteria"""
        sort_key = self.sort_var.get()
        reverse = self.reverse_var.get()

        return sorted(
            files,
            key=lambda f: self.get_file_info(os.path.join(self.current_dir, f))[sort_key],
            reverse=reverse
        )

    def get_file_info(self, file_path):
        """Get file information for sorting"""
        stat = os.stat(file_path)
        return {
            'name': os.path.basename(file_path).lower(),
            'date': stat.st_mtime,
            'size': stat.st_size,
            'type': os.path.splitext(file_path)[1].lower()
        }

    def refresh_files(self):
        """Refresh file display with current sort settings"""
        self.load_files()

    def format_file_size(self, size):
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

class TikZColorHelper:
    """Helper class for managing TikZ colors with automatic text color selection"""

    def __init__(self):
        self.color_definitions = {}
        self.default_colors = {
            'airis4d_blue': (41, 128, 185),
            'airis4d_green': (39, 174, 96),
            'airis4d_orange': (243, 156, 18),
            'airis4d_red': (231, 76, 60),
            'airis4d_purple': (155, 89, 182),
            'airis4d_teal': (26, 188, 156),
            'airis4d_gray': (149, 165, 166)
        }

    def define_color(self, name, color_value):
        """Define a custom color for TikZ"""
        self.color_definitions[name] = color_value

    def get_text_color(self, background_color_name):
        """Get optimal text color for a given background color"""
        if background_color_name in self.color_definitions:
            color_value = self.color_definitions[background_color_name]
        elif background_color_name in self.default_colors:
            color_value = self.default_colors[background_color_name]
        else:
            # Try to parse the color
            color_value = background_color_name

        return get_xor_text_color(color_value)

    def generate_tikz_color_definitions(self):
        """Generate TikZ color definition commands"""
        definitions = []

        # Add default colors
        for name, rgb in self.default_colors.items():
            definitions.append(f"\\definecolor{{{name}}}{{RGB}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}")

        # Add custom colors
        for name, color_value in self.color_definitions.items():
            if isinstance(color_value, tuple) and len(color_value) == 3:
                # RGB tuple
                definitions.append(f"\\definecolor{{{name}}}{{RGB}}{{{color_value[0]},{color_value[1]},{color_value[2]}}}")
            elif isinstance(color_value, str):
                # Color name or hex
                definitions.append(f"\\definecolor{{{name}}}{{HTML}}{{{color_value.lstrip('#')}}}")

        return "\n".join(definitions)

    def create_colored_node(self, text, background_color, node_options=""):
        """Create a TikZ node with automatic text color"""
        text_color = self.get_text_color(background_color)

        tikz_code = f"""
\\node[{node_options}, fill={background_color}, text={text_color}] {{{text}}};
"""
        return tikz_code.strip()

class ColorPickerDialog(ctk.CTkToplevel):
    """Dialog for selecting colors with preview and XOR text color feedback"""

    def __init__(self, parent, current_color=None):
        super().__init__(parent)
        self.title("TikZ Color Picker")
        self.geometry("500x400")
        self.result = None

        # Initialize color helper
        self.color_helper = TikZColorHelper()

        # Center dialog
        self.transient(parent)
        self.grab_set()

        # Create UI
        self.create_widgets(current_color)

    def create_widgets(self, current_color):
        """Create color picker widgets"""
        # Main container
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Color selection frame
        color_frame = ctk.CTkFrame(main_frame)
        color_frame.pack(fill="x", padx=5, pady=5)

        # Color name entry
        ctk.CTkLabel(color_frame, text="Color Name:").pack(side="left", padx=5)
        self.color_name_entry = ctk.CTkEntry(color_frame, width=150)
        self.color_name_entry.pack(side="left", padx=5)
        self.color_name_entry.insert(0, "custom_color")

        # RGB inputs
        rgb_frame = ctk.CTkFrame(main_frame)
        rgb_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(rgb_frame, text="R:").pack(side="left", padx=5)
        self.r_entry = ctk.CTkEntry(rgb_frame, width=50)
        self.r_entry.pack(side="left", padx=5)
        self.r_entry.insert(0, "128")

        ctk.CTkLabel(rgb_frame, text="G:").pack(side="left", padx=5)
        self.g_entry = ctk.CTkEntry(rgb_frame, width=50)
        self.g_entry.pack(side="left", padx=5)
        self.g_entry.insert(0, "128")

        ctk.CTkLabel(rgb_frame, text="B:").pack(side="left", padx=5)
        self.b_entry = ctk.CTkEntry(rgb_frame, width=50)
        self.b_entry.pack(side="left", padx=5)
        self.b_entry.insert(0, "128")

        # Preview frame
        preview_frame = ctk.CTkFrame(main_frame, height=100)
        preview_frame.pack(fill="x", padx=5, pady=10)
        preview_frame.pack_propagate(False)

        self.preview_label = ctk.CTkLabel(
            preview_frame,
            text="Sample Text",
            font=("Arial", 16, "bold"),
            width=400,
            height=80,
            corner_radius=10
        )
        self.preview_label.pack(pady=10)

        # Text color feedback
        feedback_frame = ctk.CTkFrame(main_frame)
        feedback_frame.pack(fill="x", padx=5, pady=5)

        self.feedback_label = ctk.CTkLabel(
            feedback_frame,
            text="Recommended text color: ",
            font=("Arial", 12)
        )
        self.feedback_label.pack()

        # Color buttons for quick selection
        colors_frame = ctk.CTkFrame(main_frame)
        colors_frame.pack(fill="x", padx=5, pady=10)

        ctk.CTkLabel(colors_frame, text="Quick Colors:").pack(anchor="w", padx=5, pady=5)

        quick_colors = [
            ("Blue", "airis4d_blue", "#2980b9"),
            ("Green", "airis4d_green", "#27ae60"),
            ("Orange", "airis4d_orange", "#f39c12"),
            ("Red", "airis4d_red", "#e74c3c"),
            ("Purple", "airis4d_purple", "#9b59b6"),
            ("Teal", "airis4d_teal", "#1abc9c"),
            ("Gray", "airis4d_gray", "#95a5a6")
        ]

        buttons_frame = ctk.CTkFrame(colors_frame)
        buttons_frame.pack(fill="x", padx=5, pady=5)

        for color_name, color_id, hex_color in quick_colors:
            btn = ctk.CTkButton(
                buttons_frame,
                text=color_name,
                command=lambda c=color_id, h=hex_color: self.select_quick_color(c, h),
                width=80,
                fg_color=hex_color,
                hover_color=hex_color
            )
            btn.pack(side="left", padx=2, pady=2)

        # Action buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", padx=5, pady=10)

        ctk.CTkButton(
            button_frame,
            text="Update Preview",
            command=self.update_preview
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Apply",
            command=self.apply_color
        ).pack(side="right", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.cancel
        ).pack(side="right", padx=5)

        # Set initial color if provided
        if current_color:
            self.update_preview()

    def select_quick_color(self, color_id, hex_color):
        """Select a quick color"""
        self.color_name_entry.delete(0, 'end')
        self.color_name_entry.insert(0, color_id)

        # Parse hex to RGB
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        self.r_entry.delete(0, 'end')
        self.r_entry.insert(0, str(r))

        self.g_entry.delete(0, 'end')
        self.g_entry.insert(0, str(g))

        self.b_entry.delete(0, 'end')
        self.b_entry.insert(0, str(b))

        self.update_preview()

    def update_preview(self):
        """Update color preview"""
        try:
            r = int(self.r_entry.get())
            g = int(self.g_entry.get())
            b = int(self.b_entry.get())

            # Ensure values are in range
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            hex_color = f"#{r:02x}{g:02x}{b:02x}"

            # Update preview label
            self.preview_label.configure(fg_color=hex_color)

            # Calculate and show recommended text color
            text_color = get_xor_text_color((r, g, b))
            self.feedback_label.configure(
                text=f"Recommended text color: {text_color.upper()} (XOR rule)",
                text_color="white" if text_color == "white" else "black"
            )

        except ValueError:
            # Invalid input, use default
            self.preview_label.configure(fg_color="gray")
            self.feedback_label.configure(
                text="Invalid RGB values",
                text_color="red"
            )

    def apply_color(self):
        """Apply selected color"""
        try:
            color_name = self.color_name_entry.get().strip()
            r = int(self.r_entry.get())
            g = int(self.g_entry.get())
            b = int(self.b_entry.get())

            self.result = {
                'name': color_name,
                'rgb': (r, g, b),
                'hex': f"#{r:02x}{g:02x}{b:02x}",
                'text_color': get_xor_text_color((r, g, b))
            }
            self.destroy()
        except ValueError:
            messagebox.showerror("Error", "Invalid color values")

    def cancel(self):
        """Cancel color selection"""
        self.result = None
        self.destroy()

    @staticmethod
    def pick_color(parent, current_color=None):
        """Static method to pick color"""
        dialog = ColorPickerDialog(parent, current_color)
        dialog.wait_window()
        return dialog.result

#------------------------------------------------------------------------------------------
class PreambleEditor(ctk.CTkToplevel):
    def __init__(self, parent, current_preamble=None):
        super().__init__(parent)
        self.title("Preamble Editor")
        self.geometry("800x600")

        # Store the default preamble
        self.default_preamble = get_beamer_preamble(
            "Title", "Subtitle", "Author", "Institution", "Short Inst", "\\today"
        )

        # Create UI
        self.create_editor()
        self.create_toolbar()

        # Load current preamble if provided, else load default
        if current_preamble:
            self.editor.delete('1.0', 'end')
            self.editor.insert('1.0', current_preamble)
        else:
            self.reset_to_default()

    def create_editor(self):
        """Create the preamble text editor"""
        # Editor frame
        editor_frame = ctk.CTkFrame(self)
        editor_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        # Editor with syntax highlighting
        self.editor = ctk.CTkTextbox(
            editor_frame,
            wrap="none",
            font=("Courier", 12)
        )
        self.editor.pack(fill="both", expand=True, padx=5, pady=5)

        # Add syntax highlighting
        self.syntax_highlighter = BeamerSyntaxHighlighter(self.editor)

    def create_toolbar(self):
        """Create toolbar with editor controls"""
        toolbar = ctk.CTkFrame(self)
        toolbar.pack(fill="x", padx=10, pady=5)

        # Create buttons
        buttons = [
            ("Reset to Default", self.reset_to_default),
            ("Save Custom", self.save_custom),
            ("Load Custom", self.load_custom),
            ("Apply", self.apply_changes),
            ("Cancel", self.cancel_changes)
        ]

        for text, command in buttons:
            ctk.CTkButton(
                toolbar,
                text=text,
                command=command,
                width=100
            ).pack(side="left", padx=5)

    def reset_to_default(self):
        """Reset preamble to default"""
        if messagebox.askyesno("Reset Preamble",
                             "Are you sure you want to reset to default preamble?"):
            self.editor.delete('1.0', 'end')
            self.editor.insert('1.0', self.default_preamble)
            self.syntax_highlighter.highlight()

    def save_custom(self):
        """Save current preamble as custom template"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".tex",
            filetypes=[("TeX files", "*.tex"), ("All files", "*.*")],
            title="Save Custom Preamble"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.editor.get('1.0', 'end-1c'))
                messagebox.showinfo("Success", "Custom preamble saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving preamble: {str(e)}")

    def load_custom(self):
        """Load custom preamble template"""
        file_path = filedialog.askopenfilename(
            filetypes=[("TeX files", "*.tex"), ("All files", "*.*")],
            title="Load Custom Preamble"
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.editor.delete('1.0', 'end')
                    self.editor.insert('1.0', content)
                    self.syntax_highlighter.highlight()
            except Exception as e:
                messagebox.showerror("Error", f"Error loading preamble: {str(e)}")

    def apply_changes(self):
        """Apply preamble changes and close editor"""
        self.preamble = self.editor.get('1.0', 'end-1c')
        self.destroy()

    def cancel_changes(self):
        """Cancel changes and close editor"""
        self.preamble = None
        self.destroy()

    @staticmethod
    def edit_preamble(parent, current_preamble=None):
        """Static method to handle preamble editing"""
        editor = PreambleEditor(parent, current_preamble)
        editor.wait_window()
        return editor.preamble if hasattr(editor, 'preamble') else None
#------------------------------------------------------------------------------------------
class NotesToggleFrame(ctk.CTkFrame):
    """Frame containing notes display options with tooltips"""
    def __init__(self, parent, main_editor, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Store reference to main editor
        self.main_editor = main_editor

        # Notes mode variable
        self.notes_mode = tk.StringVar(value="both")

        # Create radio buttons for different notes modes
        modes = [
            ("PDF Only", "slides", "Hide all presentation notes"),
            ("Notes Only", "notes", "Show only presentation notes"),
            ("PDF with Notes", "both", "Show PDF with notes on second screen")
        ]

        # Create label
        label = ctk.CTkLabel(self, text="Notes Display:", anchor="w")
        label.pack(side="left", padx=5)
        self.create_tooltip(label, "Select how notes should appear in the final output")

        # Create radio buttons
        for text, value, tooltip in modes:
            btn = ctk.CTkRadioButton(
                self,
                text=text,
                variable=self.notes_mode,
                value=value
            )
            btn.pack(side="left", padx=10)
            self.create_tooltip(btn, tooltip)

    def get_notes_directive(self) -> str:
        """Return the appropriate beamer directive based on current mode"""
        mode = self.notes_mode.get()
        if mode == "slides":
            return "\\setbeameroption{hide notes}"
        elif mode == "notes":
            return "\\setbeameroption{show only notes}"
        else:  # both
            return "\\setbeameroption{show notes on second screen=right}"


class TerminalIO:
    """Improved terminal I/O handler for BSG-IDE integration"""
    def __init__(self, editor):
        self.editor = editor

    def write(self, text, color="white"):
        """Write to terminal with color"""
        if hasattr(self.editor, 'terminal'):
            self.editor.terminal.write(text, color)

    def terminal_input(self, prompt):
        """Get input from terminal with proper synchronization"""
        if hasattr(self.editor, 'terminal'):
            # Use the terminal's input method directly
            text = self.editor.terminal.terminal_input(prompt)
            # Write back the input for visual feedback
            self.editor.terminal.write(text + "\n", "green")
            return text
        else:
            # Fallback to standard input
            text = input(prompt)
            return text


class InstallationDialog(ctk.CTkToplevel):
    """Installation progress dialog with visual feedback"""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("BSG-IDE Installation")
        self.geometry("400x300")

        # Make dialog modal
        self.transient(parent)
        self.grab_set()

        # Center dialog
        self.center_window()

        # Progress elements
        self.status_label = ctk.CTkLabel(self, text="Preparing installation...")
        self.status_label.pack(pady=20)

        self.progress = ctk.CTkProgressBar(self)
        self.progress.pack(pady=10, padx=20, fill="x")
        self.progress.set(0)

        self.detail_text = ctk.CTkTextbox(self, height=150)
        self.detail_text.pack(pady=10, padx=20, fill="both", expand=True)

    def center_window(self):
        """Center dialog on parent window"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'+{x}+{y}')

    def update_progress(self, value: float, message: str):
        """Update progress bar and status message"""
        self.progress.set(value)
        self.status_label.configure(text=message)
        self.detail_text.insert('end', f"{message}\n")
        self.detail_text.see('end')
        self.update()

    def write(self, message: str, color: str = None):
        """Write message to detail text with optional color"""
        if color:
            tag = f"color_{color}"
            self.detail_text.tag_configure(tag, foreground=color)
            self.detail_text.insert('end', message + "\n", tag)
        else:
            self.detail_text.insert('end', message + "\n")
        self.detail_text.see('end')
        self.update()

class InstallationManager:
    """Enhanced installation manager with GUI feedback"""
    def __init__(self, parent):
        self.parent = parent
        self.dialog = None
        self.system = platform.system()
        # Set is_admin attribute first
        self.is_admin = False  # Default value
        try:
            if self.system == "Windows":
                import ctypes
                self.is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
            else:
                self.is_admin = os.geteuid() == 0
        except Exception as e:
            print(f"Warning: Could not determine admin status: {e}")
            self.is_admin = False

        # Now get installation paths after is_admin is set
        self.install_paths = self._get_install_paths()

    def install(self):
        """Start installation process with GUI feedback"""
        try:
            self.dialog = InstallationDialog(self.parent)

            # Verify Python environment
            self.dialog.update_progress(0.1, "Checking Python environment...")
            self._verify_python_env()

            # Create installation directories
            self.dialog.update_progress(0.2, "Creating directories...")
            self._create_directories()

            # Copy files
            self.dialog.update_progress(0.4, "Copying required files...")
            self._copy_files()

            # Create launcher
            if not self.create_launcher():
                return False

            # Setup OS integration
            self.dialog.update_progress(0.6, "Setting up system integration...")
            self._setup_os_integration()

            # Install dependencies
            self.dialog.update_progress(0.8, "Installing dependencies...")
            self._install_dependencies()

            # Final checks
            self.dialog.update_progress(0.9, "Performing final checks...")
            self._verify_installation()

            # Complete
            self.dialog.update_progress(1.0, "Installation completed successfully!")
            self.dialog.write("✓ BSG-IDE installation completed!", "green")

            # Close dialog after delay
            self.dialog.after(2000, self.dialog.destroy)
            return True


        except Exception as e:
            if self.dialog:
                self.dialog.write(f"✗ Installation failed: {str(e)}", "red")
                self.dialog.write("\nDetails:", "red")
                self.dialog.write(traceback.format_exc(), "red")
                # Keep dialog open for error review
            return False

    def create_launcher(self):
        """Create platform-specific launcher scripts and shortcuts"""
        try:
            if self.system == "Linux":
                return self._create_linux_launcher()
            elif self.system == "Windows":
                return self._create_windows_launcher()
            else:  # macOS
                return self._create_macos_launcher()
        except Exception as e:
            if self.dialog:
                self.dialog.write(f"✗ Error creating launcher: {str(e)}", "red")
            else:
                print(f"✗ Error creating launcher: {str(e)}")
            return False

    def _create_linux_launcher(self):
        """Create Linux launcher script"""
        try:
            launcher_path = self.install_paths['bin'] / 'bsg-ide'
            launcher_content = f"""#!/usr/bin/env python3
    import sys
    import os
    from pathlib import Path

    # Add installation directory to Python path
    sys.path.insert(0, "{self.install_paths['base']}")

    # Import and run main program
    try:
        from BSG_IDE import main
        main()
    except Exception as e:
        print(f"Error starting BSG-IDE: {{str(e)}}")
        import traceback
        traceback.print_exc()
    """
            launcher_path.write_text(launcher_content)
            launcher_path.chmod(0o755)
            return True
        except Exception as e:
            print(f"Error creating launcher: {e}")
            return False

    def _create_windows_launcher(self):
        """Create Windows launcher script and shortcuts"""
        try:
            # Create batch file
            launcher_path = self.install_paths['bin'] / 'bsg-ide.bat'
            launcher_content = f"""@echo off
set PYTHONPATH={self.install_paths['base']};%PYTHONPATH%
python -c "from BSG_IDE import main; main()" %*
"""
            launcher_path.write_text(launcher_content)

            if self.dialog:
                self.dialog.write(f"✓ Created launcher script: {launcher_path}", "green")
            else:
                print(f"✓ Created launcher script: {launcher_path}")

            # Create Start Menu shortcut
            try:
                import winshell
                from win32com.client import Dispatch
                shortcut_dir = self.install_paths['start_menu'] / "BSG-IDE"
                shortcut_dir.mkdir(parents=True, exist_ok=True)

                shell = Dispatch('WScript.Shell')
                shortcut = shell.CreateShortCut(str(shortcut_dir / "BSG-IDE.lnk"))
                shortcut.Targetpath = str(launcher_path)
                shortcut.save()

                if self.dialog:
                    self.dialog.write("✓ Created Start Menu shortcut", "green")
                else:
                    print("✓ Created Start Menu shortcut")
            except ImportError:
                if self.dialog:
                    self.dialog.write("! Warning: Could not create Start Menu shortcut", "yellow")
                else:
                    print("! Warning: Could not create Start Menu shortcut")

            return True

        except Exception as e:
            if self.dialog:
                self.dialog.write(f"✗ Error creating Windows launcher: {str(e)}", "red")
            else:
                print(f"✗ Error creating Windows launcher: {str(e)}")
            return False

    def _create_macos_launcher(self):
        """Create macOS launcher script and application bundle"""
        try:
            # Create launcher script
            launcher_path = self.install_paths['bin'] / 'bsg-ide'
            launcher_content = f"""#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add installation directory to Python path
sys.path.insert(0, "{self.install_paths['base']}")

# Import and run main program
try:
    from BSG_IDE import main
    main()
except Exception as e:
    print(f"Error starting BSG-IDE: {{str(e)}}")
    import traceback
    traceback.print_exc()
"""
            launcher_path.write_text(launcher_content)
            launcher_path.chmod(0o755)

            if self.dialog:
                self.dialog.write(f"✓ Created launcher script: {launcher_path}", "green")
            else:
                print(f"✓ Created launcher script: {launcher_path}")

            return True

        except Exception as e:
            if self.dialog:
                self.dialog.write(f"✗ Error creating macOS launcher: {str(e)}", "red")
            else:
                print(f"✗ Error creating macOS launcher: {str(e)}")
            return False

 #------------------------------------------------------------------------------------
    def copy_resources(self, source_dir):
        """Copy required resources to installation directories"""
        try:
            # Define required files and their destinations
            required_files = {
                'BSG_IDE.py': ['base', 'python_site'],
                'BeamerSlideGenerator.py': ['base', 'python_site'],
                'requirements.txt': ['base'],
                'airis4d_logo.png': ['base', 'resources', 'share'],  # Added multiple destinations
                'bsg-ide.png': ['base', 'resources', 'share']
            }

            # Copy each required file
            for filename, destinations in required_files.items():
                source_file = source_dir / filename
                if source_file.exists():
                    for dest_type in destinations:
                        if dest_type in self.install_paths:
                            # Create destination directory if needed
                            dest_dir = self.install_paths[dest_type]
                            dest_dir.mkdir(parents=True, exist_ok=True)

                            # For resources, ensure resources subdirectory exists
                            if dest_type == 'base':
                                resources_dir = dest_dir / 'resources'
                                resources_dir.mkdir(parents=True, exist_ok=True)
                                dest_path = resources_dir / filename
                            else:
                                dest_path = dest_dir / filename

                            try:
                                shutil.copy2(source_file, dest_path)
                                if self.dialog:
                                    self.dialog.write(f"✓ Copied {filename} to {dest_path}", "green")
                                else:
                                    print(f"✓ Copied {filename} to {dest_path}")
                            except Exception as e:
                                if self.dialog:
                                    self.dialog.write(f"! Warning: Could not copy {filename} to {dest_path}: {e}", "yellow")
                                else:
                                    print(f"! Warning: Could not copy {filename} to {dest_path}: {e}")
                else:
                    if self.dialog:
                        self.dialog.write(f"! Warning: Source file {filename} not found in {source_dir}", "yellow")
                    else:
                        print(f"! Warning: Source file {filename} not found in {source_dir}")

            # Create additional resource directories if needed
            for path_type in ['resources', 'share']:
                if path_type in self.install_paths:
                    resource_dir = self.install_paths[path_type] / 'resources'
                    resource_dir.mkdir(parents=True, exist_ok=True)
                    if self.dialog:
                        self.dialog.write(f"✓ Created resource directory: {resource_dir}", "green")

            return True

        except Exception as e:
            if self.dialog:
                self.dialog.write(f"✗ Error copying resources: {str(e)}", "red")
            else:
                print(f"✗ Error copying resources: {str(e)}")
            return False


    def _get_install_paths(self):
        """Get installation paths based on platform and permissions"""
        paths = {}

        # Get user's home directory
        home_dir = Path.home()

        if self.is_admin:
            # System-wide installation paths
            if self.system == "Linux":
                paths.update({
                    'base': Path('/usr/local/lib/bsg-ide'),
                    'bin': Path('/usr/local/bin'),
                    'share': Path('/usr/local/share/bsg-ide'),
                    'resources': Path('/usr/local/share/bsg-ide/resources'),
                    'applications': Path('/usr/share/applications'),
                    'icons': Path('/usr/share/icons/hicolor'),
                    'python_site': Path(site.getsitepackages()[0]) / 'bsg_ide'
                })
            elif self.system == "Windows":
                program_files = Path(os.environ.get('PROGRAMFILES', 'C:\\Program Files'))
                paths.update({
                    'base': program_files / 'BSG-IDE',
                    'bin': program_files / 'BSG-IDE' / 'bin',
                    'share': program_files / 'BSG-IDE' / 'share',
                    'resources': program_files / 'BSG-IDE' / 'resources',
                    'start_menu': Path(os.environ['PROGRAMDATA']) / 'Microsoft/Windows/Start Menu/Programs',
                    'python_site': Path(site.getsitepackages()[0]) / 'bsg_ide'
                })
            else:  # macOS
                paths.update({
                    'base': Path('/Applications/BSG-IDE.app/Contents'),
                    'share': Path('/Applications/BSG-IDE.app/Contents/Resources'),
                    'resources': Path('/Applications/BSG-IDE.app/Contents/Resources'),
                    'bin': Path('/usr/local/bin'),
                    'python_site': Path(site.getsitepackages()[0]) / 'bsg_ide'
                })
        else:
            # User-specific installation paths
            if self.system == "Linux":
                paths.update({
                    'base': home_dir / '.local/lib/bsg-ide',
                    'bin': home_dir / '.local/bin',
                    'share': home_dir / '.local/share/bsg-ide',
                    'resources': home_dir / '.local/share/bsg-ide/resources',
                    'applications': home_dir / '.local/share/applications',
                    'icons': home_dir / '.local/share/icons/hicolor',
                    'python_site': Path(site.getusersitepackages()) / 'bsg_ide'
                })
            elif self.system == "Windows":
                appdata = Path(os.environ['APPDATA'])
                paths.update({
                    'base': appdata / 'BSG-IDE',
                    'bin': appdata / 'BSG-IDE' / 'bin',
                    'share': appdata / 'BSG-IDE' / 'share',
                    'resources': appdata / 'BSG-IDE' / 'resources',
                    'start_menu': appdata / 'Microsoft/Windows/Start Menu/Programs',
                    'python_site': Path(site.getusersitepackages()) / 'bsg_ide'
                })
            else:  # macOS
                paths.update({
                    'base': home_dir / 'Applications/BSG-IDE.app/Contents',
                    'share': home_dir / 'Library/Application Support/BSG-IDE',
                    'resources': home_dir / 'Library/Application Support/BSG-IDE/resources',
                    'bin': home_dir / '.local/bin',
                    'python_site': Path(site.getusersitepackages()) / 'bsg_ide'
                })

        return paths
 #-----------------------------------------------------------------------------------

    def create_directory_structure(self):
        """Create all required directories for installation"""
        try:
            # Create all directories from install_paths
            for path_type, path in self.install_paths.items():
                try:
                    # Skip creation of python_site as it's handled by pip
                    if path_type == 'python_site':
                        continue

                    path.mkdir(parents=True, exist_ok=True)
                    if self.dialog:
                        self.dialog.write(f"✓ Created directory: {path}", "green")
                    else:
                        print(f"✓ Created directory: {path}")
                except Exception as e:
                    if self.dialog:
                        self.dialog.write(f"! Warning: Could not create {path}: {e}", "yellow")
                    else:
                        print(f"! Warning: Could not create {path}: {e}")

            # Create additional required subdirectories
            media_dirs = [
                self.install_paths['base'] / 'media_files',
                self.install_paths['resources'] / 'templates',
                self.install_paths['resources'] / 'icons'
            ]

            for directory in media_dirs:
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    if self.dialog:
                        self.dialog.write(f"✓ Created directory: {directory}", "green")
                    else:
                        print(f"✓ Created directory: {directory}")
                except Exception as e:
                    if self.dialog:
                        self.dialog.write(f"! Warning: Could not create {directory}: {e}", "yellow")
                    else:
                        print(f"! Warning: Could not create {directory}: {e}")

            # Create .keep files in empty directories to ensure they're tracked
            for directory in self.install_paths.values():
                if directory.is_dir() and not any(directory.iterdir()):
                    keep_file = directory / '.keep'
                    try:
                        keep_file.touch()
                    except Exception:
                        pass

            return True

        except Exception as e:
            if self.dialog:
                self.dialog.write(f"✗ Error creating directory structure: {str(e)}", "red")
            else:
                print(f"✗ Error creating directory structure: {str(e)}")
            return False



    def _verify_python_env(self):
        """Verify Python environment and dependencies"""
        required_version = (3, 7)
        if sys.version_info < required_version:
            raise RuntimeError(f"Python {required_version[0]}.{required_version[1]} or higher required")

        # Check virtual environment
        if not hasattr(sys, 'real_prefix') and not hasattr(sys, 'base_prefix') != sys.prefix:
            self.dialog.write("! Creating virtual environment...")
            venv_path = os.path.join(os.path.expanduser("~"), "my_python")
            venv.create(venv_path, with_pip=True)
            self.dialog.write("✓ Virtual environment created", "green")

    def _create_directories(self):
        """Create required directories with proper permissions"""
        for path in self.install_paths.values():
            path.mkdir(parents=True, exist_ok=True)
            self.dialog.write(f"✓ Created: {path}", "green")

    def _copy_files(self):
        """Copy required files to installation directories"""
        package_root = Path(__file__).parent
        required_files = {
            'BSG_IDE.py': {'dest': ['base', 'python_site']},
            'BeamerSlideGenerator.py': {'dest': ['base', 'python_site']},
            'requirements.txt': {'dest': ['base']},
            'airis4d_logo.png': {'dest': ['resources']},
            'bsg-ide.png': {'dest': ['resources']}
        }

        for filename, config in required_files.items():
            src = package_root / filename
            if src.exists():
                for dest_key in config['dest']:
                    dest = self.install_paths[dest_key] / filename
                    shutil.copy2(src, dest)
                    self.dialog.write(f"✓ Copied {filename} to {dest}", "green")
            else:
                self.dialog.write(f"! Warning: {filename} not found", "yellow")

    def setup_os_integration(self):
       """Setup OS icons and desktop integration"""
       icon_sizes = [16, 32, 48, 64, 128, 256]

       try:
           # Create icon objects for both logos
           icons = {}
           for icon_name in ['bsg-ide.png', 'airis4d_logo.png']:
               source = self.install_paths['resources'] / icon_name
               if source.exists():
                   with Image.open(source) as img:
                       icons[icon_name] = img.copy()

           if self.system == "Linux":
               # Linux hicolor theme structure
               for size in icon_sizes:
                   icon_dir = self.install_paths['icons'] / f"{size}x{size}" / 'apps'
                   icon_dir.mkdir(parents=True, exist_ok=True)

                   for icon_name, img in icons.items():
                       resized = img.copy()
                       resized.thumbnail((size, size), Image.Resampling.LANCZOS)
                       resized.save(icon_dir / icon_name)

               subprocess.run(['gtk-update-icon-cache', '--force', '--quiet',
                             str(self.install_paths['icons'])], check=False)

           elif self.system == "Windows":
               # Windows ICO format with multiple sizes
               for icon_name, img in icons.items():
                   ico_sizes = [(16,16), (32,32), (48,48), (256,256)]
                   ico_path = self.install_paths['resources'] / f"{icon_name[:-4]}.ico"

                   ico_images = []
                   for size in ico_sizes:
                       resized = img.copy()
                       resized.thumbnail(size, Image.Resampling.LANCZOS)
                       ico_images.append(resized)

                   ico_images[0].save(ico_path, format='ICO', sizes=ico_sizes, append_images=ico_images[1:])

                   # Associate icon with application
                   import winreg
                   with winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, "BSG-IDE") as key:
                       winreg.SetValue(key, "DefaultIcon", winreg.REG_SZ, str(ico_path))

           elif self.system == "Darwin":  # macOS
               # ICNS format for macOS
               for icon_name, img in icons.items():
                   icns_path = self.install_paths['resources'] / f"{icon_name[:-4]}.icns"

                   icns_sizes = [16, 32, 128, 256, 512]
                   icns_images = []

                   for size in icns_sizes:
                       resized = img.copy()
                       resized.thumbnail((size, size), Image.Resampling.LANCZOS)
                       icns_images.append(resized)

                   icns_images[0].save(icns_path, format='ICNS', append_images=icns_images[1:])

                   # Copy to app bundle
                   bundle_icons = self.install_paths['base'] / 'Contents' / 'Resources'
                   bundle_icons.mkdir(parents=True, exist_ok=True)
                   shutil.copy2(icns_path, bundle_icons)

           print("✓ Icons installed successfully")
           return True

       except Exception as e:
           print(f"Error setting up OS integration: {str(e)}")
           return False

    def _setup_linux_icons(self, icon_src):
        """Setup Linux icon themes"""
        sizes = [16, 32, 48, 64, 128, 256]

        for size in sizes:
            icon_dir = self.install_paths['icons'] / f"{size}x{size}" / 'apps'
            icon_dir.mkdir(parents=True, exist_ok=True)

            # Resize and save icon
            with Image.open(icon_src) as img:
                resized = img.resize((size, size), Image.Resampling.LANCZOS)
                resized.save(icon_dir / 'bsg-ide.png')

    def _setup_windows_icons(self, icon_src):
        """Setup Windows icons"""
        import win32api
        import win32con

        # Create .ico file
        ico_path = self.install_paths['resources'] / 'bsg-ide.ico'
        img = Image.open(icon_src)
        img.save(ico_path, format='ICO', sizes=[(16,16), (32,32), (48,48), (256,256)])

        # Associate icon with file types
        file_types = ['.bsg', '.tex']
        for ext in file_types:
            win32api.RegSetValue(
                win32con.HKEY_CLASSES_ROOT,
                f"BSG-IDE{ext}\\DefaultIcon",
                win32con.REG_SZ,
                str(ico_path)
            )

    def _setup_macos_icons(self, icon_src):
        """Setup macOS icons"""
        # Create .icns file for macOS
        icns_path = self.install_paths['resources'] / 'bsg-ide.icns'

        with Image.open(icon_src) as img:
            # macOS icon sizes
            sizes = [16, 32, 128, 256, 512]
            icons = []
            for size in sizes:
                icons.append(img.resize((size, size), Image.Resampling.LANCZOS))

            # Save as icns
            icons[0].save(icns_path, format='ICNS', append_images=icons[1:])

    def _install_dependencies(self):
        """Install Python package dependencies"""
        requirements_file = self.install_paths['base'] / 'requirements.txt'
        if requirements_file.exists():
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "-r", str(requirements_file)
                ])
                self.dialog.write("✓ Installed Python dependencies", "green")
            except subprocess.CalledProcessError as e:
                self.dialog.write(f"! Warning: Some dependencies failed to install: {e}", "yellow")

    def _verify_installation(self):
        """Verify installation success"""
        problems = []

        # Check core files
        for path in self.install_paths.values():
            if not path.exists():
                problems.append(f"Missing directory: {path}")

        # Check dependencies
        try:
            import customtkinter
            import PIL
            self.dialog.write("✓ Required packages verified", "green")
        except ImportError as e:
            problems.append(f"Missing dependency: {e}")

        if problems:
            self.dialog.write("\nInstallation warnings:", "yellow")
            for problem in problems:
                self.dialog.write(f"! {problem}", "yellow")


#------------------------------------------------------------------------------------------
class SimpleRedirector:
    """Output redirector"""
    def __init__(self, terminal, color="white"):
        self.terminal = terminal
        self.color = color

    def write(self, text):
        if text.strip():
            self.terminal.write(text, self.color)

    def flush(self):
        pass


#------------------------------------------------------------------------------------------
class BeamerSyntaxHighlighter:
    """Syntax highlighting for Beamer/LaTeX content"""

    def __init__(self, text_widget: ctk.CTkTextbox):
        self.ctk_text = text_widget
        self.text = text_widget._textbox
        self.active = True

        # Create fonts
        self.normal_font = tk.font.Font(family="TkFixedFont")
        self.italic_font = tk.font.Font(family="TkFixedFont", slant="italic")

        # Define syntax highlighting colors
        self.colors = {
            'command': '#FF6B6B',     # LaTeX commands
            'media': '#4ECDC4',       # Media directives
            'bullet': '#95A5A6',      # Bullet points
            'url': '#45B7D1',         # URLs
            'bracket': '#FFB86C',     # Curly brackets
            'comment': '#6272A4',     # Comments
            'rgb': '#50FA7B',         # RGB color commands
            'textcolor': '#BD93F9' ,   # textcolor commands
            'tikz':'#8A2BE2'  # Blue violet for TikZ
        }

        # Configure tags on the underlying Text widget
        for tag, color in self.colors.items():
            self.text.tag_configure(tag, foreground=color, font=self.normal_font)

        # Special formatting for comments with italic font
        self.text.tag_configure("comment",
                              foreground=self.colors['comment'],
                              font=self.italic_font)

        # Define syntax patterns
        self.patterns = [
            (r'\\[a-zA-Z]+', 'command'),
            (r'\\(file|play|None)\s', 'media'),
            (r'^-\s.*$', 'bullet'),
            (r'https?://\S+', 'url'),
            (r'\{.*?\}', 'bracket'),
            (r'%.*$', 'comment'),
            (r'\\textcolor\{.*?\}', 'textcolor'),
            (r'\[RGB\]\{[^\}]*\}', 'rgb'),
            (r'\\begin\{tikzpicture\}', 'tikz'),  # TikZ start
            (r'\\end\{tikzpicture\}', 'tikz'),    # TikZ end
            (r'\\draw.*', 'tikz'),                # TikZ draw commands
            (r'\\node.*', 'tikz'),                # TikZ nodes
            (r'\\fill.*', 'tikz'),                # TikZ fill commands
        ]

        # Bind events to the CTkTextbox
        self.ctk_text.bind('<KeyRelease>', self.highlight)
        self.ctk_text.bind('<Control-v>', lambda e: self.after_paste())
        # Initialize presentation metadata
        self.presentation_info = {
            'title': '',
            'subtitle': '',
            'author': '',
            'institution': 'Artificial Intelligence Research and Intelligent Systems (airis4D)',
            'short_institute': 'airis4D',
            'date': '\\today'
        }


    def toggle(self) -> None:
        """Toggle syntax highlighting on/off"""
        self.active = not self.active
        if self.active:
            self.highlight()
        else:
            self.clear_highlighting()

    def clear_highlighting(self) -> None:
        """Remove all highlighting"""
        for tag in self.colors.keys():
            self.text.tag_remove(tag, "1.0", "end")

    def highlight(self, event=None) -> None:
        """Apply syntax highlighting to the text"""
        if not self.active:
            return

        self.clear_highlighting()
        for pattern, tag in self.patterns:
            self.highlight_pattern(pattern, tag)

    def highlight_pattern(self, pattern: str, tag: str) -> None:
        """Apply highlighting for a specific pattern"""
        content = self.text.get("1.0", "end-1c")
        lines = content.split('\n')

        for line_num, line in enumerate(lines, start=1):
            for match in re.finditer(pattern, line):
                start = match.start()
                end = match.end()
                start_index = f"{line_num}.{start}"
                end_index = f"{line_num}.{end}"
                self.text.tag_add(tag, start_index, end_index)

    def after_paste(self) -> None:
        """Handle highlighting after paste operation"""
        self.text.after(10, self.highlight)
        # Also trigger spell checking
        if hasattr(self.ctk_text.master, 'spell_checking_enabled') and self.ctk_text.master.spell_checking_enabled:
            self.ctk_text.master.check_spelling()

class LaTeXErrorEditor(ctk.CTkToplevel):
    """Interactive editor for fixing LaTeX compilation errors"""

    def __init__(self, parent, tex_file_path, error_line_num, error_message, error_context, log_content=None):
        # Handle different parent types gracefully
        # If parent is a CTk window with 'root' attribute, use that
        # Otherwise use parent directly
        if hasattr(parent, 'root') and isinstance(parent.root, (ctk.CTk, tk.Tk)):
            actual_parent = parent.root
        elif hasattr(parent, 'winfo_toplevel'):
            actual_parent = parent
        else:
            actual_parent = parent

        super().__init__(actual_parent)

        self.tex_file_path = tex_file_path
        self.error_line_num = error_line_num
        self.error_message = error_message
        self.error_context = error_context
        self.log_content = log_content
        self.result = None  # 'fixed', 'skip', 'abort'
        self.fixed_content = None
        self.modified = False
        self.original_content = None

        self.title(f"LaTeX Error Editor - Line {error_line_num}")
        self.geometry("1000x700")

        # Make dialog modal - safe handling
        try:
            self.transient(actual_parent)
            self.grab_set()
        except:
            pass

        # Center window
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 1000) // 2
        y = (self.winfo_screenheight() - 700) // 2
        self.geometry(f"+{x}+{y}")

        self.create_widgets()
        self.load_file_at_error_line()

    def create_widgets(self):
        """Create the editor UI"""

        # Main container
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Error info frame
        error_frame = ctk.CTkFrame(main_frame, fg_color="#8B0000")
        error_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            error_frame,
            text="⚠ ERROR DETECTED",
            font=("Arial", 14, "bold"),
            text_color="white"
        ).pack(anchor="w", padx=10, pady=5)

        ctk.CTkLabel(
            error_frame,
            text=self.error_message[:200],
            font=("Arial", 11),
            text_color="#FFB6C1",
            wraplength=950
        ).pack(anchor="w", padx=10, pady=5)

        ctk.CTkLabel(
            error_frame,
            text=f"Location: Line {self.error_line_num}",
            font=("Arial", 10),
            text_color="#FFD700"
        ).pack(anchor="w", padx=10, pady=5)

        # Editor frame with line numbers
        editor_frame = ctk.CTkFrame(main_frame)
        editor_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create a paned window for side-by-side view
        paned = tk.PanedWindow(editor_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        paned.pack(fill="both", expand=True)

        # Left side - Code editor with line numbers
        left_frame = tk.Frame(paned)
        paned.add(left_frame, width=500)

        # Line number area
        line_number_frame = tk.Frame(left_frame, bg="#1e1e1e", width=50)
        line_number_frame.pack(side="left", fill="y")
        line_number_frame.pack_propagate(False)

        self.line_numbers = tk.Text(
            line_number_frame,
            width=4,
            padx=3,
            takefocus=0,
            border=0,
            background="#1e1e1e",
            foreground="#858585",
            font=("Courier", 11),
            wrap="none",
            state="disabled"
        )
        self.line_numbers.pack(side="left", fill="y")

        # Main text editor
        editor_container = tk.Frame(left_frame)
        editor_container.pack(side="left", fill="both", expand=True)

        # Scrollbar
        scrollbar = tk.Scrollbar(editor_container)
        scrollbar.pack(side="right", fill="y")

        self.editor = tk.Text(
            editor_container,
            font=("Courier", 11),
            background="#1e1e1e",
            foreground="#d4d4d4",
            insertbackground="white",
            wrap="none",
            yscrollcommand=scrollbar.set,
            tabs=("4c", "8c", "12c", "16c", "20c", "24c")
        )
        self.editor.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.on_scroll)

        # Configure tags for syntax highlighting
        self.editor.tag_config("error_line", background="#8B0000", foreground="white")
        self.editor.tag_config("error_highlight", background="#FF4444", foreground="white")
        self.editor.tag_config("line_number", foreground="#858585")
        self.editor.tag_config("commented_line", background="#2d2d2d", foreground="#6a9955")

        # Bind events
        self.editor.bind("<KeyRelease>", self.on_text_change)
        self.editor.bind("<MouseWheel>", self.on_mousewheel)
        self.editor.bind("<Button-4>", self.on_mousewheel)
        self.editor.bind("<Button-5>", self.on_mousewheel)

        # Right side - Error context and suggestions
        right_frame = ctk.CTkFrame(paned)
        paned.add(right_frame, width=400)

        # Error details
        ctk.CTkLabel(
            right_frame,
            text="Error Details",
            font=("Arial", 12, "bold")
        ).pack(anchor="w", padx=10, pady=5)

        error_detail_text = ctk.CTkTextbox(right_frame, height=150, font=("Courier", 10))
        error_detail_text.pack(fill="x", padx=10, pady=5)
        error_detail_text.insert("1.0", self.error_context[:1000])
        error_detail_text.configure(state="disabled")

        # Common fixes suggestions
        ctk.CTkLabel(
            right_frame,
            text="Suggested Fixes",
            font=("Arial", 12, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))

        self.suggestions = self.generate_suggestions()
        suggestions_text = ctk.CTkTextbox(right_frame, height=150, font=("Courier", 10))
        suggestions_text.pack(fill="both", expand=True, padx=10, pady=5)
        suggestions_text.insert("1.0", self.suggestions)
        suggestions_text.configure(state="disabled")

        # Quick fix buttons
        quick_fix_frame = ctk.CTkFrame(right_frame)
        quick_fix_frame.pack(fill="x", padx=10, pady=10)

        quick_fixes = [
            ("Remove Empty []", self.remove_empty_brackets),
            ("Escape # as \\#", self.escape_hash),
            ("Fix Itemize", self.fix_itemize),
            ("Comment Out Line", self.comment_out_line),
            ("Restore Original", self.restore_original)
        ]

        for text, command in quick_fixes:
            ctk.CTkButton(
                quick_fix_frame,
                text=text,
                command=command,
                width=110,
                height=30
            ).pack(side="left", padx=2, pady=5)

        # Bottom buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", padx=5, pady=10)

        ctk.CTkButton(
            button_frame,
            text="✓ Apply Fix & Recompile",
            command=self.apply_and_recompile,
            width=150,
            fg_color="#28a745",
            hover_color="#218838"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="⏭ Skip This Error",
            command=self.skip_error,
            width=120,
            fg_color="#ffc107",
            hover_color="#e0a800",
            text_color="black"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="⏹ Abort Compilation",
            command=self.abort_compilation,
            width=120,
            fg_color="#dc3545",
            hover_color="#c82333"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="📋 Copy Error",
            command=self.copy_error,
            width=100
        ).pack(side="right", padx=5)

        # Status bar
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Edit the code on the left, then click 'Apply Fix & Recompile'",
            font=("Arial", 10),
            text_color="#888888"
        )
        self.status_label.pack(fill="x", padx=5, pady=5)

    def load_file_at_error_line(self):
        """Load the tex file and highlight the error line"""
        try:
            with open(self.tex_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            self.original_content = ''.join(lines)

            # Insert into editor
            for i, line in enumerate(lines, 1):
                self.editor.insert("end", line)
                self.update_line_numbers()

            # Highlight the error line
            if self.error_line_num and 1 <= self.error_line_num <= len(lines):
                error_start = f"{self.error_line_num}.0"
                error_end = f"{self.error_line_num}.end"
                self.editor.tag_add("error_line", error_start, error_end)
                self.editor.see(error_start)

        except Exception as e:
            self.editor.insert("1.0", f"% Error loading file: {e}\n")

    def update_line_numbers(self):
        """Update line number display"""
        lines = self.editor.get("1.0", "end-1c").split('\n')
        line_numbers = '\n'.join(str(i) for i in range(1, len(lines) + 1))
        self.line_numbers.configure(state="normal")
        self.line_numbers.delete("1.0", "end")
        self.line_numbers.insert("1.0", line_numbers)
        self.line_numbers.configure(state="disabled")

    def on_scroll(self, *args):
        """Sync scrolling between editor and line numbers"""
        self.editor.yview(*args)
        self.line_numbers.yview(*args)

    def on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -1 * (event.delta // 120)

        self.editor.yview_scroll(delta, "units")
        self.line_numbers.yview_scroll(delta, "units")
        return "break"

    def on_text_change(self, event=None):
        """Update line numbers when text changes"""
        self.update_line_numbers()
        self.modified = True
        self.editor.tag_remove("error_line", "1.0", "end")
        self.editor.tag_remove("error_highlight", "1.0", "end")
        self.status_label.configure(text="Changes made - Click 'Apply Fix & Recompile' to save")

    def generate_suggestions(self):
        """Generate suggested fixes based on error message"""
        suggestions = []

        if 'Illegal parameter number' in self.error_message:
            suggestions.append("• Use \\# instead of # in regular text")
            suggestions.append("• In nested macros, use ## instead of #")
        elif 'Missing number' in self.error_message:
            suggestions.append("• Check for empty [] brackets and remove them")
            suggestions.append("• Look for standalone numbers that should be part of commands")
        elif 'Undefined control sequence' in self.error_message:
            suggestions.append("• Check for misspelled LaTeX commands")
            suggestions.append("• Make sure required packages are loaded")
        elif 'Environment' in self.error_message and 'undefined' in self.error_message:
            suggestions.append("• Check if the environment name is spelled correctly")
            suggestions.append("• Make sure the required package is loaded")
        else:
            suggestions.append("• Check for unmatched braces {}")
            suggestions.append("• Look for special characters that need escaping (# $ % & _ { } ~ ^)")
            suggestions.append("• Check for missing \\item commands in itemize/enumerate")

        suggestions.append("\nQuick fixes available in the buttons below:")
        suggestions.append("  • 'Remove Empty []' - removes problematic empty brackets")
        suggestions.append("  • 'Escape # as \\#' - escapes hash symbols")
        suggestions.append("  • 'Fix Itemize' - fixes malformed itemize environments")
        suggestions.append("  • 'Comment Out Line' - comments out the problematic line (preserves it)")
        suggestions.append("  • 'Restore Original' - reverts all changes")

        return '\n'.join(suggestions)

    def remove_empty_brackets(self):
        """Remove empty brackets from the error line"""
        if not self.error_line_num:
            return
        line_content = self.editor.get(f"{self.error_line_num}.0", f"{self.error_line_num}.end")
        fixed = line_content.replace('[]', '').replace('[,]', '')
        self.editor.delete(f"{self.error_line_num}.0", f"{self.error_line_num}.end")
        self.editor.insert(f"{self.error_line_num}.0", fixed)
        self.modified = True
        self.status_label.configure(text="Removed empty brackets")
        self.update_line_numbers()

    def escape_hash(self):
        """Escape # characters in the error line"""
        if not self.error_line_num:
            return
        line_content = self.editor.get(f"{self.error_line_num}.0", f"{self.error_line_num}.end")
        fixed = line_content.replace('#', '\\#')
        self.editor.delete(f"{self.error_line_num}.0", f"{self.error_line_num}.end")
        self.editor.insert(f"{self.error_line_num}.0", fixed)
        self.modified = True
        self.status_label.configure(text="Escaped # characters")
        self.update_line_numbers()

    def fix_itemize(self):
        """Fix malformed itemize environments"""
        content = self.editor.get("1.0", "end-1c")
        lines = content.split('\n')
        fixed_lines = []
        in_itemize = False

        for line in lines:
            if '\\begin{itemize}' in line:
                in_itemize = True
            elif '\\end{itemize}' in line:
                in_itemize = False

            if in_itemize and re.match(r'^\s*\d+\.?\d*\s*$', line):
                fixed_lines.append(f'% {line}  % Commented out by fix_itemize')
                continue

            fixed_lines.append(line)

        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", '\n'.join(fixed_lines))
        self.modified = True
        self.status_label.configure(text="Fixed itemize environment")
        self.update_line_numbers()

    def comment_out_line(self):
        """Comment out the problematic line"""
        if not self.error_line_num:
            return
        line_content = self.editor.get(f"{self.error_line_num}.0", f"{self.error_line_num}.end")

        if not line_content.strip().startswith('%'):
            fixed = f'% {line_content}  % Commented out by error editor'
            self.editor.delete(f"{self.error_line_num}.0", f"{self.error_line_num}.end")
            self.editor.insert(f"{self.error_line_num}.0", fixed)
            self.modified = True
            self.status_label.configure(text=f"Commented out line {self.error_line_num}")
            self.editor.tag_add("commented_line", f"{self.error_line_num}.0", f"{self.error_line_num}.end")
        else:
            self.status_label.configure(text="Line already commented out")
        self.update_line_numbers()

    def restore_original(self):
        """Restore original content"""
        if self.original_content:
            self.editor.delete("1.0", "end")
            self.editor.insert("1.0", self.original_content)
            self.modified = True
            self.status_label.configure(text="Restored original content")
            self.update_line_numbers()
            if self.error_line_num:
                self.editor.tag_add("error_line", f"{self.error_line_num}.0", f"{self.error_line_num}.end")

    def copy_error(self):
        """Copy error to clipboard"""
        self.clipboard_clear()
        self.clipboard_append(self.error_message)
        self.status_label.configure(text="Error copied to clipboard")

    def save_changes(self):
        """Save changes to file"""
        try:
            content = self.editor.get("1.0", "end-1c")
            temp_file = self.tex_file_path + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content)
            shutil.move(temp_file, self.tex_file_path)
            self.fixed_content = content
            self.status_label.configure(text="✓ Changes saved successfully!")
            return True
        except Exception as e:
            self.status_label.configure(text=f"✗ Error saving: {e}")
            return False

    def apply_and_recompile(self):
        """Apply changes and recompile"""
        if self.save_changes():
            self.result = 'fixed'
            self.destroy()

    def skip_error(self):
        """Skip this error"""
        self.result = 'skip'
        self.destroy()

    def abort_compilation(self):
        """Abort compilation"""
        self.result = 'abort'
        self.destroy()


class BeamerSlideEditor(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Version and info - KEEP ORIGINAL
        AIRIS4D_ASCII_LOGO = """
        /\\
       /  \\   airis4D
      / /\\ \\
     /_/  \\_\\ LABS
    """
        self.__version__ = "4.6.4"
        self.__author__ = "Ninan Sajeeth Philip"
        self.__license__ = "Creative Commons"
        self.logo_ascii = AIRIS4D_ASCII_LOGO

        self.add_tikz_color_helper()

        # Initialize dictionary for notes buttons - KEEP ORIGINAL
        self.notes_buttons = {}

        # Initialize paths - KEEP ORIGINAL
        self.package_root, self.resources_dir = setup_paths()

        # Initialize logo before creating widgets - KEEP ORIGINAL but reordered
        self.has_logo = False
        self.logo_image = None
        self.setup_logo()  # Moved earlier for proper initialization

        # Rest of initialization... - PRESERVE ALL ORIGINAL SETUP
        self.create_widgets()

        # Create terminal I/O interface - KEEP ORIGINAL
        self.terminal_io = TerminalIO(self)

        # Initialize session manager with error handling - KEEP ORIGINAL
        try:
            self.session_manager = SessionManager()
            self.session_data = self.session_manager.load_session()
        except Exception as e:
            print(f"Warning: Session management unavailable: {str(e)}")
            self.session_manager = None

            documents_dir = None
            try:
                possible_docs = [
                    Path.home() / 'Documents',
                    Path.home() / 'documents',
                    Path(os.path.expandvars('%USERPROFILE%\\Documents'))
                ]
                for doc_path in possible_docs:
                    if doc_path.exists() and doc_path.is_dir():
                        documents_dir = doc_path
                        break
            except:
                pass

            self.session_data = {
                'last_file': None,
                'working_directory': str(Path.cwd()),
                'recent_files': [],
                'window_size': {'width': 1200, 'height': 800},
                'window_position': {'x': None, 'y': None}
            }

        # Configure window based on session data - KEEP ORIGINAL
        self.title("BeamerSlide Generator IDE")
        self.geometry(f"{self.session_data['window_size']['width']}x{self.session_data['window_size']['height']}")
        if all(v is not None for v in self.session_data['window_position'].values()):
            self.geometry(f"+{self.session_data['window_position']['x']}+{self.session_data['window_position']['y']}")

        # Change to working directory if valid - KEEP ORIGINAL
        try:
            working_dir = Path(self.session_data['working_directory'])
            if working_dir.exists():
                os.chdir(working_dir)
                print(f"Working directory set to: {working_dir}")
            else:
                # Try to use Documents folder
                documents_dir = None
                possible_docs = [
                    Path.home() / 'Documents',
                    Path.home() / 'documents',
                    Path(os.path.expandvars('%USERPROFILE%\\Documents'))
                ]
                for doc_path in possible_docs:
                    if doc_path.exists() and doc_path.is_dir():
                        documents_dir = doc_path
                        os.chdir(documents_dir)
                        print(f"Working directory set to Documents: {documents_dir}")
                        break
                else:
                    # Fall back to home directory
                    os.chdir(Path.home())
                    print(f"Working directory set to home: {Path.home()}")
        except Exception as e:
            print(f"Warning: Could not set working directory: {str(e)}")
            print("Falling back to current directory")

        # Set the terminal I/O in BeamerSlideGenerator - KEEP ORIGINAL
        from BeamerSlideGenerator import set_terminal_io
        set_terminal_io(self.terminal_io)

        # Initialize presentation metadata - KEEP ORIGINAL
        self.presentation_info = {
            'title': '',
            'subtitle': '',
            'author': '',
            'institution': 'Artificial Intelligence Research and Intelligent Systems (airis4D)',
            'short_institute': 'airis4D',
            'date': '\\today'
        }

        # Configure grid - KEEP ORIGINAL
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Initialize variables - KEEP ORIGINAL
        self.current_file = None
        self.slides = []
        self.current_slide_index = -1

        # Setup keyboard shortcuts - KEEP ORIGINAL
        self.setup_keyboard_shortcuts()

        # Setup Python paths - KEEP ORIGINAL
        setup_python_paths()

        # Adjust grid weights to accommodate terminal - KEEP ORIGINAL
        self.grid_rowconfigure(1, weight=3)  # Main editor
        self.grid_rowconfigure(4, weight=1)  # Terminal

        # Setup output redirection after terminal creation - KEEP ORIGINAL
        self.setup_output_redirection()
        self.use_enhanced_tooltips=True
        # Change to working directory if valid - KEEP ORIGINAL (duplicate in original)
        try:
            if self.session_data['working_directory']:
                os.chdir(self.session_data['working_directory'])
        except Exception as e:
            print(f"Warning: Could not change to saved working directory: {str(e)}")

        # Load last file if it exists - KEEP ORIGINAL
        if self.session_data['last_file'] and os.path.exists(self.session_data['last_file']):
            self.after(100, lambda: self.load_file(self.session_data['last_file']))

        # Bind window events - KEEP ORIGINAL
        self.bind('<Configure>', self.on_window_configure)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize spell checking - KEEP ORIGINAL but ensure it runs after UI creation
        self.setup_spellchecking()

        # Add binding to close context menu - KEEP ORIGINAL
        self.bind("<Button-1>", self.hide_spelling_menu)

        # === ENHANCED FEATURES INITIALIZATION ===

        # Initialize enhanced LaTeX help system
        self.setup_enhanced_latex_help()

        # Initialize enhanced tooltip system
        self.setup_enhanced_tooltips()

        # Auto-create first slide - KEEP ORIGINAL
        self.auto_create_first_slide()

        # Initialize Grammarly features - MODIFIED: Delay binding until after UI is ready
        self.grammarly = GrammarlyIntegration(self)
        self.setup_grammarly_integration_delayed()

        self.setup_automated_grammarly()

        # Add command index button to toolbar - KEEP ORIGINAL
        self.enhance_toolbar()

        # DELAYED: Bind Grammarly to editors (after they are definitely created)
        self.after(100, self.setup_grammarly_bindings)

        # NEW: Auto-prompt for Grammarly setup on first run (optional)
        self.after(500, self.auto_prompt_grammarly_setup)

        # Initialize enhanced features
        self.enhanced_command_index = None

        # Initialize autocomplete system (replaces old setup_autocomplete calls)
        self.autocomplete_system = IntelligentAutocomplete(self)

        # Setup enhanced features after UI is ready
        self.after(150, self.initialize_enhanced_features)

        # Add undo/redo stacks for slide operations
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_history = 50  # Limit history size

        # Add a flag for tracking deleted slides visually
        self.show_deleted_slides = True

        self.setup_line_mask_context_menu()

        # Add a status label to show current context
        self.status_label = ctk.CTkLabel(
            self,
            text="Ready | Ctrl+Delete: Mask slide (in slide list) or mask line (in editor)",
            font=("Arial", 10),
            text_color="#888888",
            height=20
        )
        self.status_label.grid(row=5, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 5))

        self.content_editor._textbox.bind('<Button-1>', self.on_line_click_to_unmask)
        self.notes_editor._textbox.bind('<Button-1>', self.on_line_click_to_unmask)

        # Call this after loading a file to debug
        self.debug_slide_data()

    def mask_slide(self, slide_index: int) -> bool:
        """
        Mask a slide by prepending % to all its content items.
        Returns True if successful, False otherwise.
        """
        if not 0 <= slide_index < len(self.slides):
            return False

        # Don't mask already masked slides
        if self.is_slide_masked(slide_index):
            return False

        # Create a deep copy for undo history
        original_slide = self._deep_copy_slide(self.slides[slide_index])

        # Mask the slide content
        masked_slide = self._create_masked_slide(self.slides[slide_index])

        # Save to undo stack BEFORE modification
        self._push_to_undo_stack({
            'action': 'mask',
            'index': slide_index,
            'original': original_slide,
            'masked': masked_slide
        })

        # Apply masking
        self.slides[slide_index] = masked_slide

        # Clear redo stack after new action
        self.redo_stack.clear()

        # Update UI
        self.update_slide_list()
        if self.current_slide_index == slide_index:
            self.load_slide(slide_index)

        self.write(f"✓ Slide {slide_index + 1} masked (marked as deleted)\n", "yellow")
        return True

    def unmask_slide(self, slide_index: int) -> bool:
        """
        Restore a masked slide to its original state.
        Returns True if successful, False otherwise.
        """
        if not 0 <= slide_index < len(self.slides):
            return False

        # Only unmask if it's actually masked
        if not self.is_slide_masked(slide_index):
            return False

        # Create a deep copy for undo history
        current_slide = self._deep_copy_slide(self.slides[slide_index])

        # Restore original content by removing comment markers
        restored_slide = self._restore_from_masked_slide(current_slide)

        # Save to undo stack
        self._push_to_undo_stack({
            'action': 'unmask',
            'index': slide_index,
            'original': current_slide,
            'restored': restored_slide
        })

        # Apply restoration
        self.slides[slide_index] = restored_slide

        # Clear redo stack
        self.redo_stack.clear()

        # Update UI
        self.update_slide_list()
        if self.current_slide_index == slide_index:
            self.load_slide(slide_index)

        self.write(f"✓ Slide {slide_index + 1} restored\n", "green")
        return True

    def delete_slide(self) -> None:
        """Modified delete_slide that masks instead of deleting and advances to next slide"""
        if self.current_slide_index >= 0 and self.current_slide_index < len(self.slides):
            if self.is_slide_masked(self.current_slide_index):
                # Ask for confirmation for permanent deletion of already masked slide
                if messagebox.askyesno("Permanent Delete",
                                       f"Slide {self.current_slide_index + 1} is already masked.\n\n"
                                       "Do you want to permanently delete it?\n"
                                       "This action cannot be undone with Ctrl+Z."):
                    deleted_index = self.current_slide_index
                    self._permanently_delete_slide(deleted_index)
                    self.write(f"✓ Slide {deleted_index + 1} permanently deleted\n", "red")

                    # After permanent deletion, move to appropriate slide
                    if self.slides:
                        if deleted_index >= len(self.slides):
                            self.current_slide_index = len(self.slides) - 1
                        else:
                            self.current_slide_index = deleted_index
                        self.load_slide(self.current_slide_index)
                    else:
                        self.clear_editor()
            else:
                # Mask the slide and advance
                self.mask_slide(self.current_slide_index)
                self.write(f"✓ Slide {self.current_slide_index + 1} masked (marked as deleted)\n", "yellow")

                # Advance to next slide
                if self.current_slide_index + 1 < len(self.slides):
                    self.current_slide_index += 1
                    self.load_slide(self.current_slide_index)
                    self.write(f"→ Advanced to slide {self.current_slide_index + 1}\n", "cyan")
                elif self.current_slide_index > 0:
                    self.current_slide_index -= 1
                    self.load_slide(self.current_slide_index)
                    self.write(f"← Reached end, moved to previous slide {self.current_slide_index + 1}\n", "cyan")

                self.update_slide_list()
                self.highlight_current_slide()
        else:
            messagebox.showwarning("Warning", "No slide to delete!")

    def _permanently_delete_slide(self, index: int) -> None:
        """Permanently remove a slide and update navigation"""
        if 0 <= index < len(self.slides):
            # Save to undo stack for potential restoration
            self._push_to_undo_stack({
                'action': 'permanent_delete',
                'index': index,
                'slide': self._deep_copy_slide(self.slides[index])
            })

            del self.slides[index]

            # Update current slide index after deletion
            if self.slides:
                if index >= len(self.slides):
                    # Deleted the last slide, move to previous
                    self.current_slide_index = len(self.slides) - 1
                else:
                    # Stay at same index (next slide shifts into place)
                    self.current_slide_index = index
            else:
                self.current_slide_index = -1

            self.redo_stack.clear()
            self.update_slide_list()

            if self.current_slide_index >= 0:
                self.load_slide(self.current_slide_index)
            else:
                self.clear_editor()

            self.write(f"✓ Slide {index + 1} permanently deleted\n", "red")

    def undo(self) -> None:
        """Undo the last slide operation (mask/unmask/delete)"""
        if not self.undo_stack:
            self.write("Nothing to undo\n", "yellow")
            return

        action = self.undo_stack.pop()

        # Save current state to redo stack
        current_state = self._get_current_slide_state(action['index']) if 'index' in action else None

        if action['action'] == 'mask':
            # Restore original slide
            self.slides[action['index']] = self._deep_copy_slide(action['original'])
            self._push_to_redo_stack({
                'action': 'unmask',
                'index': action['index'],
                'original': action['masked'],
                'restored': action['original']
            })
            self.write(f"↩ Undo: Restored slide {action['index'] + 1}\n", "cyan")

        elif action['action'] == 'unmask':
            # Re-mask the slide
            self.slides[action['index']] = self._deep_copy_slide(action['original'])
            self._push_to_redo_stack({
                'action': 'mask',
                'index': action['index'],
                'original': action['restored'],
                'masked': action['original']
            })
            self.write(f"↩ Undo: Re-masked slide {action['index'] + 1}\n", "cyan")

        elif action['action'] == 'permanent_delete':
            # Restore permanently deleted slide
            self.slides.insert(action['index'], self._deep_copy_slide(action['slide']))
            self._push_to_redo_stack({
                'action': 'permanent_delete_redo',
                'index': action['index'],
                'slide': action['slide']
            })
            self.write(f"↩ Undo: Restored permanently deleted slide {action['index'] + 1}\n", "cyan")

        # Update UI
        if 'index' in action:
            self.current_slide_index = action['index']
        self.update_slide_list()
        if self.current_slide_index >= 0:
            self.load_slide(self.current_slide_index)
        self.highlight_current_slide()

    def redo(self) -> None:
        """Redo a previously undone slide operation"""
        if not self.redo_stack:
            self.write("Nothing to redo\n", "yellow")
            return

        action = self.redo_stack.pop()

        if action['action'] == 'mask':
            self.mask_slide(action['index'])
        elif action['action'] == 'unmask':
            self.unmask_slide(action['index'])
        elif action['action'] == 'permanent_delete_redo':
            self._permanently_delete_slide(action['index'])

        self.write(f"↪ Redo: Applied {action['action']} on slide {action['index'] + 1}\n", "cyan")

    def is_slide_masked(self, slide_index: int) -> bool:
        """Check if a slide is masked (deleted)"""
        if not 0 <= slide_index < len(self.slides):
            return False

        slide = self.slides[slide_index]

        # Check if content items are commented
        content_masked = all(
            item.strip().startswith('%') or not item.strip()
            for item in slide.get('content', [])
        )

        # Check if notes are commented
        notes_masked = all(
            note.strip().startswith('%') or not note.strip()
            for note in slide.get('notes', [])
        )

        # A slide is considered masked if all its content is commented
        return content_masked and (not slide.get('notes') or notes_masked)

    def _create_masked_slide(self, slide: dict) -> dict:
        """Create a masked version of a slide"""
        masked_slide = self._deep_copy_slide(slide)

        # Mask content items
        masked_slide['content'] = [
            f"% {item}" if item.strip() and not item.strip().startswith('%') else item
            for item in masked_slide.get('content', [])
        ]

        # Mask notes
        if 'notes' in masked_slide:
            masked_slide['notes'] = [
                f"% {note}" if note.strip() and not note.strip().startswith('%') else note
                for note in masked_slide.get('notes', [])
            ]

        # Add a marker to the title if not already marked
        if not masked_slide['title'].startswith('[DELETED]'):
            masked_slide['title'] = f"[DELETED] {masked_slide['title']}"

        return masked_slide

    def _restore_from_masked_slide(self, masked_slide: dict) -> dict:
        """Restore a masked slide to its original state"""
        restored_slide = self._deep_copy_slide(masked_slide)

        # Remove comment markers from content
        restored_slide['content'] = [
            item[2:] if item.strip().startswith('%') else item
            for item in restored_slide.get('content', [])
        ]

        # Remove comment markers from notes
        if 'notes' in restored_slide:
            restored_slide['notes'] = [
                note[2:] if note.strip().startswith('%') else note
                for note in restored_slide.get('notes', [])
            ]

        # Remove [DELETED] marker from title
        if restored_slide['title'].startswith('[DELETED]'):
            restored_slide['title'] = restored_slide['title'][10:].strip()

        return restored_slide

    def _deep_copy_slide(self, slide: dict) -> dict:
        """Create a deep copy of a slide dictionary"""
        return {
            'title': slide.get('title', ''),
            'media': slide.get('media', ''),
            'content': slide.get('content', []).copy(),
            'notes': slide.get('notes', []).copy()
        }

    def _push_to_undo_stack(self, action: dict) -> None:
        """Push an action to the undo stack with size limit"""
        self.undo_stack.append(action)
        if len(self.undo_stack) > self.max_undo_history:
            self.undo_stack.pop(0)

    def _push_to_redo_stack(self, action: dict) -> None:
        """Push an action to the redo stack with size limit"""
        self.redo_stack.append(action)
        if len(self.redo_stack) > self.max_undo_history:
            self.redo_stack.pop(0)

    def _get_current_slide_state(self, index: int) -> dict:
        """Get current state of a slide for redo stack"""
        if 0 <= index < len(self.slides):
            return self._deep_copy_slide(self.slides[index])
        return None

    def update_slide_list(self):
        """Modified update_slide_list to show masked slides differently with safe tag handling"""
        # Clear the slide list first
        self.slide_list.delete('1.0', 'end')

        # Remove any existing tags to avoid conflicts (safe to do)
        try:
            for tag in ['current_slide', 'deleted_slide', 'deleted_current', 'normal_slide']:
                self.slide_list.tag_delete(tag)
        except:
            pass  # Tags might not exist yet

        # Configure tags fresh each time
        self.slide_list.tag_config('current_slide', background='#2F3542', foreground='#FFFFFF')
        self.slide_list.tag_config('deleted_slide', foreground='#888888')
        self.slide_list.tag_config('deleted_current', background='#2F3542', foreground='#888888')
        self.slide_list.tag_config('normal_slide', foreground='#FFFFFF')

        # Add each slide to the list
        for i, slide in enumerate(self.slides):
            is_current = (i == self.current_slide_index)
            prefix = "→ " if is_current else "  "

            title = slide.get('title', 'Untitled')
            is_masked = self.is_slide_masked(i)

            if is_masked:
                # Clean title by removing [DELETED] prefix for cleaner display
                display_title = title.replace('[DELETED]', '').strip()
                if not display_title:
                    display_title = "Untitled"

                # Format the line for deleted slide
                line_text = f"{prefix}🗑 DELETED: {display_title}\n"
                self.slide_list.insert('end', line_text)

                # Apply appropriate tag based on whether it's current
                line_start = f"{i+1}.0"
                line_end = f"{i+1}.end"
                if is_current:
                    self.slide_list.tag_add('deleted_current', line_start, line_end)
                else:
                    self.slide_list.tag_add('deleted_slide', line_start, line_end)
            else:
                # Normal slide
                media_type = " [None]" if not slide.get('media') or slide.get('media') == "\\None" else ""
                line_text = f"{prefix}Slide {i+1}: {title}{media_type}\n"
                self.slide_list.insert('end', line_text)

                # Apply tag for normal slide (only if not current, current handled separately)
                if not is_current:
                    line_start = f"{i+1}.0"
                    line_end = f"{i+1}.end"
                    self.slide_list.tag_add('normal_slide', line_start, line_end)

        # Highlight current slide separately
        self.highlight_current_slide()

    def highlight_current_slide(self):
        """Highlight current slide in list with proper error handling"""
        # Safety check: make sure slide_list exists and has content
        if not hasattr(self, 'slide_list') or not self.slide_list:
            return

        # Remove previous highlights safely
        try:
            self.slide_list.tag_remove('current_slide', '1.0', 'end')
            self.slide_list.tag_remove('deleted_current', '1.0', 'end')
        except:
            pass  # Tags might not exist

        # Add highlight to current slide if valid
        if self.current_slide_index >= 0 and self.current_slide_index < len(self.slides):
            try:
                start = f"{self.current_slide_index + 1}.0"
                end = f"{self.current_slide_index + 1}.end"

                # Check if the line exists
                if float(self.slide_list.index('end-1c')) >= self.current_slide_index + 1:
                    # Check if current slide is masked
                    if self.is_slide_masked(self.current_slide_index):
                        self.slide_list.tag_add('deleted_current', start, end)
                    else:
                        self.slide_list.tag_add('current_slide', start, end)

                    # Ensure the current slide is visible
                    self.slide_list.see(start)
            except Exception as e:
                print(f"Warning: Could not highlight current slide: {e}")

    def setup_keyboard_shortcuts(self) -> None:
        """Enhanced keyboard shortcuts with undo/redo and line masking"""
        # Existing shortcuts
        self.bind('<Control-n>', lambda e: self.new_slide())
        self.bind('<Control-i>', lambda e: self.insert_slide_below())
        self.bind('<Control-d>', lambda e: self.duplicate_slide())
        self.bind('<Control-s>', lambda e: self.save_file())

        # Context-sensitive Ctrl+Delete
        self.bind('<Control-Delete>', self.handle_ctrl_delete)

        # Undo/Redo shortcuts
        self.bind('<Control-z>', lambda e: self.undo())
        self.bind('<Control-y>', lambda e: self.redo())
        self.bind('<Control-Shift-Z>', lambda e: self.redo())

        # Restore deleted slide shortcut (Shift+Ctrl+R)
        self.bind('<Control-Shift-R>', lambda e: self.restore_deleted_slide())

        # Navigation shortcuts
        self.bind('<Control-Up>', lambda e: self.move_slide(-1))
        self.bind('<Control-Down>', lambda e: self.move_slide(1))

        # Quick navigation after masking (optional)
        self.bind('<Control-Right>', lambda e: self.navigate_to_next_slide())
        self.bind('<Control-Left>', lambda e: self.navigate_to_previous_slide())

        # Delete key for slide deletion when in slide list
        self.bind('<Delete>', self.handle_delete_key)

    def navigate_to_next_slide(self):
        """Navigate to next slide"""
        if self.current_slide_index + 1 < len(self.slides):
            self.save_current_slide()
            self.current_slide_index += 1
            self.load_slide(self.current_slide_index)
            self.update_slide_list()
            self.highlight_current_slide()
            self.slide_list.see(f"{self.current_slide_index + 1}.0")
            self.write(f"→ Navigated to slide {self.current_slide_index + 1}\n", "cyan")

    def navigate_to_previous_slide(self):
        """Navigate to previous slide"""
        if self.current_slide_index > 0:
            self.save_current_slide()
            self.current_slide_index -= 1
            self.load_slide(self.current_slide_index)
            self.update_slide_list()
            self.highlight_current_slide()
            self.slide_list.see(f"{self.current_slide_index + 1}.0")
            self.write(f"← Navigated to slide {self.current_slide_index + 1}\n", "cyan")

    def restore_deleted_slide(self) -> None:
        """Restore the currently selected slide if it's deleted"""
        if self.current_slide_index >= 0 and self.is_slide_masked(self.current_slide_index):
            self.unmask_slide(self.current_slide_index)
        else:
            self.write("Current slide is not deleted\n", "yellow")

    def restore_all_deleted_slides(self) -> None:
        """Restore all masked (deleted) slides"""
        restored_count = 0
        for i in range(len(self.slides) - 1, -1, -1):
            if self.is_slide_masked(i):
                self.unmask_slide(i)
                restored_count += 1

        if restored_count > 0:
            self.write(f"✓ Restored {restored_count} deleted slide(s)\n", "green")
        else:
            self.write("No deleted slides to restore\n", "yellow")

    def permanently_delete_masked_slides(self) -> None:
        """Permanently delete all masked slides"""
        if not any(self.is_slide_masked(i) for i in range(len(self.slides))):
            self.write("No deleted slides to permanently remove\n", "yellow")
            return

        if messagebox.askyesno("Confirm",
                               "This will permanently delete all masked slides.\n"
                               "This action cannot be undone.\n\n"
                               "Continue?"):

            # Create new list without masked slides
            new_slides = []
            for slide in self.slides:
                if not self.is_slide_masked(len(new_slides)):
                    new_slides.append(slide)

            removed_count = len(self.slides) - len(new_slides)
            self.slides = new_slides

            # Adjust current slide index
            if self.current_slide_index >= len(self.slides):
                self.current_slide_index = len(self.slides) - 1

            self.update_slide_list()
            if self.current_slide_index >= 0:
                self.load_slide(self.current_slide_index)

            self.write(f"✓ Permanently removed {removed_count} slide(s)\n", "red")

    def handle_ctrl_delete(self, event):
        """Context-sensitive handler for Ctrl+Delete"""
        focused_widget = self.focus_get()

        # Update status to show current context
        if focused_widget == self.slide_list._textbox:
            self.status_label.configure(text="Ctrl+Delete: Masking current slide...", text_color="#4ECDC4")
            self.after(2000, lambda: self.status_label.configure(text="Ready | Ctrl+Delete: Mask slide (in slide list) or mask line (in editor)", text_color="#888888"))
            self.mask_current_slide()
            return "break"

        elif focused_widget == self.content_editor._textbox:
            self.status_label.configure(text="Ctrl+Delete: Masking/unmasking current line in content editor...", text_color="#4ECDC4")
            self.after(2000, lambda: self.status_label.configure(text="Ready | Ctrl+Delete: Mask slide (in slide list) or mask line (in editor)", text_color="#888888"))
            self.mask_line_in_editor(event)
            return "break"

        elif focused_widget == self.notes_editor._textbox:
            self.status_label.configure(text="Ctrl+Delete: Masking/unmasking current line in notes editor...", text_color="#4ECDC4")
            self.after(2000, lambda: self.status_label.configure(text="Ready | Ctrl+Delete: Mask slide (in slide list) or mask line (in editor)", text_color="#888888"))
            self.mask_line_in_editor(event)
            return "break"

        elif focused_widget == self.media_entry:
            self.status_label.configure(text="Ctrl+Delete: Masking/unmasking media entry...", text_color="#4ECDC4")
            self.after(2000, lambda: self.status_label.configure(text="Ready | Ctrl+Delete: Mask slide (in slide list) or mask line (in editor)", text_color="#888888"))
            self.mask_media_entry()
            return "break"

        self.mask_current_slide()
        return "break"

    def handle_delete_key(self, event):
        """Handle plain Delete key - context-sensitive"""
        focused_widget = self.focus_get()

        # In slide list: delete/mask slide
        if focused_widget == self.slide_list._textbox:
            self.delete_slide()
            return "break"

        # In editors: let normal delete behavior happen
        return None

    def mask_current_slide(self):
        """Mask the currently selected slide and advance to next slide"""
        if self.current_slide_index >= 0 and self.current_slide_index < len(self.slides):
            if self.is_slide_masked(self.current_slide_index):
                # If already masked, ask if user wants to permanently delete
                if messagebox.askyesno("Delete Slide",
                                       f"Slide {self.current_slide_index + 1} is already masked.\n\n"
                                       "Do you want to permanently delete it?\n"
                                       "This action cannot be undone with Ctrl+Z."):
                    # Store the index before deletion
                    deleted_index = self.current_slide_index
                    self._permanently_delete_slide(deleted_index)
                    self.write(f"✓ Slide {deleted_index + 1} permanently deleted\n", "red")

                    # After permanent deletion, move to appropriate slide
                    if self.slides:
                        # If we deleted the last slide, move to previous
                        if deleted_index >= len(self.slides):
                            self.current_slide_index = len(self.slides) - 1
                        else:
                            # Otherwise stay at same index (next slide shifts into place)
                            self.current_slide_index = deleted_index
                        self.load_slide(self.current_slide_index)
                    else:
                        self.clear_editor()
            else:
                # Mask the slide
                self.mask_slide(self.current_slide_index)
                self.write(f"✓ Slide {self.current_slide_index + 1} masked (marked as deleted)\n", "yellow")

                # ADVANCE TO NEXT SLIDE AFTER MASKING
                if self.current_slide_index + 1 < len(self.slides):
                    # Move to next slide
                    self.current_slide_index += 1
                    self.load_slide(self.current_slide_index)
                    self.update_slide_list()
                    self.write(f"→ Advanced to slide {self.current_slide_index + 1}\n", "cyan")
                elif self.current_slide_index > 0:
                    # If this was the last slide, move to previous slide
                    self.current_slide_index -= 1
                    self.load_slide(self.current_slide_index)
                    self.update_slide_list()
                    self.write(f"← Reached end, moved to previous slide {self.current_slide_index + 1}\n", "cyan")
                else:
                    # Only one slide exists
                    self.update_slide_list()
                    self.write("ℹ No more slides to advance to\n", "yellow")

                # Ensure the current slide is highlighted in the list
                self.highlight_current_slide()
                self.slide_list.see(f"{self.current_slide_index + 1}.0")
        else:
            self.write("No slide to mask/delete\n", "yellow")

    def mask_media_entry(self):
        """Mask/unmask the media entry line"""
        current_media = self.media_entry.get()

        if current_media.startswith('%'):
            # Unmask: remove the % prefix
            unmasked = current_media.lstrip('%').lstrip()
            self.media_entry.delete(0, 'end')
            self.media_entry.insert(0, unmasked)
            self.write("✓ Media entry unmasked\n", "green")
            # Update the slide's media_masked flag
            if 0 <= self.current_slide_index < len(self.slides):
                self.slides[self.current_slide_index]['_media_masked'] = False
        else:
            # Mask: add % at beginning
            self.media_entry.delete(0, 'end')
            self.media_entry.insert(0, f"% {current_media}")
            self.write("✓ Media entry masked\n", "yellow")
            # Update the slide's media_masked flag
            if 0 <= self.current_slide_index < len(self.slides):
                self.slides[self.current_slide_index]['_media_masked'] = True

        # Save the change
        self.save_current_slide()

    def setup_enhanced_latex_help(self):
        """Setup enhanced LaTeX help using only EnhancedCommandDialog"""
        try:
            if ENHANCED_FEATURES_AVAILABLE:
                from EnhancedCommandDialog import LatexCommandHelper
                self.command_helper = LatexCommandHelper()
                self.use_enhanced_latex = True
                print("✓ Enhanced LaTeX help system loaded from EnhancedCommandDialog")
            else:
                # Fallback to basic implementation
                from Grammarly import LatexCommandHelper
                self.command_helper = LatexCommandHelper()
                self.use_enhanced_latex = False
                print("Using basic LaTeX help system")
        except ImportError as e:
            print(f"Error setting up LaTeX help: {e}")
            self.setup_basic_latex_help()

    def setup_enhanced_tooltips(self):
        """Setup enhanced command tooltips using only EnhancedCommandDialog"""
        try:
            if ENHANCED_FEATURES_AVAILABLE:
                from EnhancedCommandDialog import CommandTooltip
                self.tooltip_manager = CommandTooltip(self)
                self.use_enhanced_tooltips = True
                print("✓ Enhanced command tooltips loaded from EnhancedCommandDialog")
            else:
                # Fallback to basic implementation
                from Grammarly import CommandTooltip
                self.tooltip_manager = CommandTooltip(self)
                self.use_enhanced_tooltips = False
                print("Using basic command tooltips")
        except ImportError as e:
            print(f"Error setting up tooltips: {e}")
            self.tooltip_manager = None

    def show_enhanced_command_index(self):
        """Show the enhanced command index - simplified version"""
        try:
            if ENHANCED_FEATURES_AVAILABLE:
                from EnhancedCommandDialog import EnhancedCommandIndexDialog
                self.enhanced_command_index = EnhancedCommandIndexDialog(self)

                def on_closed():
                    if hasattr(self.enhanced_command_index, 'selected_command'):
                        command = self.enhanced_command_index.selected_command
                        if command:
                            self.insert_command_into_editor(command)
                    self.enhanced_command_index = None

                self.enhanced_command_index.protocol("WM_DELETE_WINDOW", on_closed)
            else:
                self.show_fallback_command_reference()
        except Exception as e:
            print(f"Error showing command index: {e}")
            self.show_fallback_command_reference()

    def initialize_enhanced_features(self):
        """Initialize enhanced features after UI is ready"""
        # Setup enhanced autocomplete
        self.setup_autocomplete()

        # Setup enhanced tooltips
        self.setup_command_tooltips()

    def setup_autocomplete(self):
        """Setup autocomplete system for editors"""
        self.autocomplete_system.setup_autocomplete(self.content_editor._textbox)
        self.autocomplete_system.setup_autocomplete(self.notes_editor._textbox)

    def setup_command_tooltips(self):
        """Setup command tooltip bindings for editors"""
        # Enhanced tooltips handle their own bindings automatically
        if self.use_enhanced_tooltips:
            print("Enhanced tooltip system active")
        else:
            # Fallback to basic tooltip setup
            self.setup_editor_tooltips(self.content_editor._textbox)
            self.setup_editor_tooltips(self.notes_editor._textbox)
            self.setup_entry_tooltips(self.media_entry)
#-----------------Init ends ----------------------------

    def setup_basic_latex_help(self):
        """Fallback to basic LaTeX help"""
        self.command_helper = LatexCommandHelper()
        self.autocomplete_system = None
        print("Using basic LaTeX help system")

    def setup_line_mask_context_menu(self):
        """Setup context menu for line masking in editors"""
        self.line_mask_menu = tk.Menu(self, tearoff=0)
        self.line_mask_menu.add_command(label="Mask/Unmask Line (Ctrl+Delete)",
                                        command=lambda: self.mask_line_in_editor())
        self.line_mask_menu.add_separator()
        self.line_mask_menu.add_command(label="Mask Multiple Lines...",
                                        command=self.mask_multiple_lines_dialog)

        # Bind to editors
        for editor in [self.content_editor._textbox, self.notes_editor._textbox]:
            editor.bind("<Button-3>", self.show_line_mask_menu)

    def show_line_mask_menu(self, event):
        """Show line mask context menu"""
        self.line_mask_menu.tk_popup(event.x_root, event.y_root)

    def mask_multiple_lines_dialog(self):
        """Dialog to mask multiple lines at once"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Mask Multiple Lines")
        dialog.geometry("400x300")
        dialog.transient(self)

        ctk.CTkLabel(dialog, text="Mask lines from:").pack(pady=5)
        start_entry = ctk.CTkEntry(dialog, width=100)
        start_entry.pack(pady=5)
        start_entry.insert(0, "1")

        ctk.CTkLabel(dialog, text="to:").pack(pady=5)
        end_entry = ctk.CTkEntry(dialog, width=100)
        end_entry.pack(pady=5)

        def apply_mask():
            try:
                start = int(start_entry.get())
                end = int(end_entry.get())

                # Get focused editor
                focused = self.focus_get()
                editor = None
                if focused in [self.content_editor._textbox, self.notes_editor._textbox]:
                    editor = focused

                if editor:
                    for line_num in range(start, end + 1):
                        line_start = editor.index(f"{line_num}.0")
                        line_end = editor.index(f"{line_num}.end")
                        line_content = editor.get(line_start, line_end)

                        if not line_content.lstrip().startswith('%'):
                            editor.delete(line_start, line_end)
                            editor.insert(line_start, f"% {line_content}")

                    self.save_current_slide()
                    self.write(f"✓ Masked lines {start}-{end}\n", "yellow")
                    dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Error masking lines: {str(e)}")

        ctk.CTkButton(dialog, text="Apply", command=apply_mask).pack(pady=20)

    def insert_command_into_editor(self, command_data):
        """Insert selected command into current editor using enhanced system"""
        focused_widget = self.focus_get()

        if focused_widget == self.content_editor._textbox:
            editor = self.content_editor
        elif focused_widget == self.notes_editor._textbox:
            editor = self.notes_editor
        else:
            editor = self.content_editor

        # Use enhanced command data if available
        if isinstance(command_data, dict) and 'example' in command_data:
            editor.insert("insert", command_data['example'] + "\n")
        else:
            editor.insert("insert", command_data + "\n")

    def setup_automated_grammarly(self):
        """Setup automated Grammarly integration"""
        self.auto_grammarly = AutomatedGrammarlyIntegration(self)

        # Check if Grammarly is already configured
        config_file = Path.home() / '.bsg-ide' / 'grammarly_config.json'
        if not config_file.exists():
            # Auto-prompt for setup after a delay
            self.after(2000, self.auto_prompt_grammarly)

    def auto_prompt_grammarly(self):
        """Automatically prompt for Grammarly setup"""
        if not self.grammarly.grammarly_enabled:
            response = messagebox.askyesno(
                "Grammarly Auto-Setup",
                "Would you like to set up Grammarly integration automatically?\n\n"
                "We'll guide you through the process with browser automation."
            )
            if response:
                self.auto_grammarly.automated_setup()

    def auto_prompt_grammarly_setup(self):
        """Automatically prompt for Grammarly setup if not configured"""
        if not self.grammarly.grammarly_enabled and not self.grammarly.grammarly_api_key:
            response = messagebox.askyesno(
                "Grammarly Integration",
                "Would you like to set up Grammarly integration for advanced grammar checking?\n\n"
                "This will open a setup dialog where you can enter your Grammarly API key."
            )
            if response:
                self.toggle_grammarly()  # This will show the setup dialog

    def add_grammarly_setup_option(self):
        """Add Grammarly setup option to help menu or settings"""
        # Add to your existing menu setup
        # Or create a dedicated "Setup Grammarly" button

        setup_button = ctk.CTkButton(
            self.toolbar,
            text="Setup Grammarly",
            command=self.setup_grammarly_manually,
            width=120,
            fg_color="#17a2b8",
            hover_color="#138496"
        )
        setup_button.pack(side="left", padx=5)
        self.create_tooltip(setup_button, "Set up Grammarly integration")

    def setup_grammarly_manually(self):
        """Manual Grammarly setup trigger"""
        if not self.grammarly.grammarly_enabled:
            self.grammarly.show_grammarly_dialog()
        else:
            messagebox.showinfo("Grammarly", "Grammarly is already enabled and configured.")

    def setup_grammarly_integration_delayed(self):
        """Setup Grammarly integration without immediate binding"""
        # Create Grammarly context menu
        self.grammarly_menu = tk.Menu(self, tearoff=0)
        self.grammarly_menu.add_command(label="Apply Suggestion", command=self.apply_grammarly_suggestion)
        self.grammarly_menu.add_command(label="Ignore Issue", command=self.ignore_grammarly_issue)
        self.grammarly_menu.add_separator()
        self.grammarly_menu.add_command(label="Explain Issue", command=self.explain_grammarly_issue)

    def setup_grammarly_bindings(self):
        """Setup Grammarly bindings after UI is fully created"""
        try:
            # Bind to both editors - NOW they should exist
            if hasattr(self, 'content_editor') and hasattr(self, 'notes_editor'):
                # Bind right-click to Grammarly menu
                for widget in [self.content_editor._textbox, self.notes_editor._textbox]:
                    widget.bind("<Button-3>", self.show_grammarly_menu)

                # Setup real-time Grammarly checking with debouncing
                def check_grammarly(event=None):
                    if hasattr(self, '_grammarly_timer'):
                        self.after_cancel(self._grammarly_timer)
                    self._grammarly_timer = self.after(1000, self.perform_grammarly_check)

                # Bind key release events
                self.content_editor._textbox.bind('<KeyRelease>', check_grammarly)
                self.notes_editor._textbox.bind('<KeyRelease>', check_grammarly)

            else:
                print("Warning: Editors not available for Grammarly binding")

        except Exception as e:
            print(f"Warning: Could not setup Grammarly bindings: {e}")


#---------------Help Windows -----------------------------
    def auto_create_first_slide(self):
        """Automatically create first slide on startup"""
        # Create initial slide if none exists
        if not hasattr(self, 'slides') or not self.slides:
            self.slides = [{
                'title': 'Presentation Title',
                'media': '',
                'content': ['- First bullet point'],
                'notes': ['• Speaker notes for first slide']
            }]
            self.current_slide_index = 0
            self.load_slide(0)
            self.update_slide_list()

    def setup_editor_tooltips(self, text_widget):
        """Setup tooltips for text widgets"""
        text_widget.bind('<Motion>', self.on_editor_motion)
        text_widget.bind('<Leave>', self.on_editor_leave)

    def setup_entry_tooltips(self, entry_widget):
        """Setup tooltips for entry widgets"""
        entry_widget.bind('<Motion>', self.on_entry_motion)
        entry_widget.bind('<Leave>', self.on_entry_leave)

    def on_editor_motion(self, event):
        """Handle mouse motion in editor for command detection"""
        widget = event.widget
        index = widget.index(f"@{event.x},{event.y}")

        # Get the word under cursor
        word_start = widget.index(f"{index} wordstart")
        word_end = widget.index(f"{index} wordend")
        word = widget.get(word_start, word_end)

        # Check if it's a LaTeX command
        if word.startswith('\\') and len(word) > 1:
            self.show_command_tooltip(widget, word, event.x_root, event.y_root)
        else:
            self.tooltip_manager.hide_tooltip()

    def on_entry_motion(self, event):
        """Handle mouse motion in entry for command detection"""
        widget = event.widget
        content = widget.get()

        # Simple check for commands in entry
        if '\\' in content:
            # Find command near cursor position
            cursor_pos = widget.index(tk.INSERT)
            text_before = content[:cursor_pos]
            if '\\' in text_before:
                last_slash = text_before.rfind('\\')
                next_space = content.find(' ', last_slash)
                if next_space == -1:
                    next_space = len(content)

                command = content[last_slash:next_space]
                if command.startswith('\\') and len(command) > 1:
                    self.show_command_tooltip(widget, command, event.x_root, event.y_root)
                    return

        self.tooltip_manager.hide_tooltip()

    def on_editor_leave(self, event):
        """Handle mouse leaving editor"""
        self.tooltip_manager.hide_tooltip()

    def on_entry_leave(self, event):
        """Handle mouse leaving entry"""
        self.tooltip_manager.hide_tooltip()

    def show_command_tooltip(self, widget, command, x, y):
        """Show tooltip for LaTeX command"""
        self.tooltip_manager.show_tooltip(widget, command, x, y)
#---------------Help Windows Ends Grammarly starts-------------------------
    def enhance_toolbar(self):
        """Add new features to toolbar"""
        # Add command index button
        self.command_index_button = ctk.CTkButton(
            self.toolbar,
            text="Enhanced Command Index",
            command=self.show_enhanced_command_index,  # Use enhanced version
            width=140,
            fg_color="#6f42c1",
            hover_color="#5a2d91"
        )
        self.command_index_button.pack(side="left", padx=5)

        # Add Grammarly button - FIXED: Use correct initial state
        grammarly_status = "On" if self.grammarly.grammarly_enabled else "Off"
        grammarly_color = "#28a745" if self.grammarly.grammarly_enabled else "#dc3545"

        self.grammarly_button = ctk.CTkButton(
            self.toolbar,
            text=f"Grammarly: {grammarly_status}",
            command=self.toggle_grammarly,  # Ensure this points to the correct method
            width=120,
            fg_color=grammarly_color,
            hover_color="#a71e2a"
        )
        self.grammarly_button.pack(side="left", padx=5)

        # Update tooltips
        self.create_tooltip(self.command_index_button, "Show complete command reference with examples")
        self.create_tooltip(self.grammarly_button, "Enable/disable Grammarly grammar checking")

        # Add automated Grammarly setup button
        self.auto_grammarly_button = ctk.CTkButton(
            self.toolbar,
            text="🚀 Auto-Setup Grammarly",
            command=self.auto_grammarly.automated_setup,
            width=140,
            fg_color="#FF6B35",
            hover_color="#E55A2B"
        )
        self.auto_grammarly_button.pack(side="left", padx=5)
        self.create_tooltip(self.auto_grammarly_button,
                           "Automated Grammarly setup with browser guidance")

    def show_fallback_command_reference(self):
        """Fallback command reference if dialog fails"""
        simple_commands = """
    Basic LaTeX Commands Quick Reference:

    Document Structure:
    \\title{Presentation Title} - Sets presentation title
    \\author{Author Name} - Sets author name
    \\institute{Institution} - Sets institution name
    \\date{\\today} - Sets presentation date

    Slide Content:
    \\file{media_files/image.png} - Insert image
    \\play{media_files/video.mp4} - Embed video
    - Bullet point - Creates a bullet point

    Text Formatting:
    \\textcolor{red}{Text} - Colors text
    \\textbf{Bold Text} - Bold text
    \\textit{Italic Text} - Italic text

    For complete reference, check the full Command Index.
    """

        messagebox.showinfo("LaTeX Commands Quick Reference", simple_commands)

    def insert_command_into_editor(self, command_data):
        """Insert selected command into current editor"""
        # Determine which editor has focus
        focused_widget = self.focus_get()

        if focused_widget == self.content_editor._textbox:
            editor = self.content_editor
        elif focused_widget == self.notes_editor._textbox:
            editor = self.notes_editor
        else:
            editor = self.content_editor  # Default to content editor

        # Insert the command example
        editor.insert("insert", command_data['example'] + "\n")

    def toggle_grammarly(self):
        """Toggle Grammarly integration with proper setup dialog"""
        if not self.grammarly.grammarly_enabled:
            # Show setup dialog when enabling Grammarly
            self.grammarly.show_grammarly_dialog()
        else:
            # Disable Grammarly
            self.grammarly.grammarly_enabled = False
            self.grammarly.save_grammarly_settings()
            self.grammarly_button.configure(text="Grammarly: Off", fg_color="#dc3545")
            self.write("Grammarly disabled\n", "yellow")

        # Update button state based on current status
        if self.grammarly.grammarly_enabled:
            self.grammarly_button.configure(text="Grammarly: On", fg_color="#28a745")
            self.setup_realtime_grammarly()

    def setup_realtime_grammarly(self):
        """Setup real-time Grammarly checking"""
        if not self.grammarly.grammarly_enabled:
            return

    # Bind to content changes with debouncing
    def check_grammarly(event=None):
        if hasattr(self, '_grammarly_timer'):
            self.after_cancel(self._grammarly_timer)

        self._grammarly_timer = self.after(1000, self.perform_grammarly_check)


    def perform_grammarly_check(self):
        """Perform Grammarly check on current text"""
        if not self.grammarly.grammarly_enabled:
            return

        # Get text from focused editor
        focused_widget = self.focus_get()

        if focused_widget in [self.content_editor._textbox, self.notes_editor._textbox]:
            text = focused_widget.get("1.0", "end-1c")

            if len(text.strip()) > 10:  # Only check substantial text
                suggestions = self.grammarly.check_text_grammarly(text)
                if suggestions:
                    self.grammarly.apply_grammarly_suggestions(focused_widget, suggestions)

    def setup_grammarly_integration(self):
        """Setup Grammarly integration UI"""
        # Create Grammarly context menu
        self.grammarly_menu = tk.Menu(self, tearoff=0)
        self.grammarly_menu.add_command(label="Apply Suggestion", command=self.apply_grammarly_suggestion)
        self.grammarly_menu.add_command(label="Ignore Issue", command=self.ignore_grammarly_issue)
        self.grammarly_menu.add_separator()
        self.grammarly_menu.add_command(label="Explain Issue", command=self.explain_grammarly_issue)

        # Bind right-click to Grammarly menu
        for widget in [self.content_editor._textbox, self.notes_editor._textbox]:
            widget.bind("<Button-3>", self.show_grammarly_menu)

    def show_grammarly_menu(self, event):
        """Show Grammarly context menu"""
        if not self.grammarly.grammarly_enabled:
            return

        widget = event.widget
        index = widget.index(f"@{event.x},{event.y}")

        # Check if click is on a Grammarly issue
        tags = widget.tag_names(index)
        if "grammarly_issue" in tags:
            self.current_grammarly_issue = index
            self.grammarly_menu.tk_popup(event.x_root, event.y_root)

    def apply_grammarly_suggestion(self):
        """Apply selected Grammarly suggestion"""
        # Implementation for applying suggestions
        pass

    def ignore_grammarly_issue(self):
        """Ignore Grammarly issue"""
        pass

    def explain_grammarly_issue(self):
        """Explain Grammarly issue"""
        pass

#---------------------------------------Grammarly Ends--------------------------------

    def extract_presentation_info_from_tex(self, content: str) -> None:
        """Extract presentation metadata from TeX content"""
        import re

        # Patterns for extracting presentation info
        patterns = {
            'title': r'\\title{([^}]*)}',
            'subtitle': r'\\subtitle{([^}]*)}',
            'author': r'\\author{([^}]*)}',
            'institute': r'\\institute{([^}]*)}',
            'date': r'\\date{([^}]*)}'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                # Clean up LaTeX formatting
                value = match.group(1)
                value = re.sub(r'\\textcolor{[^}]*}{([^}]*)}', r'\1', value)
                value = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', value)
                self.presentation_info[key] = value.strip()

        # Extract logo if present
        logo_match = re.search(r'\\logo{([^}]*)}', content)
        if logo_match:
            logo_content = logo_match.group(1)
            # Extract image path from includegraphics
            img_match = re.search(r'\\includegraphics(?:\[[^\]]*\])?{([^}]*)}', logo_content)
            if img_match:
                self.presentation_info['logo'] = img_match.group(1)

    def enhanced_extract_slides_from_tex(self, content: str) -> list:
        """Enhanced slide extraction with comprehensive LaTeX feature support"""
        slides = []
        import re

        # First isolate the document body
        doc_match = re.search(r'\\begin{document}(.*?)\\end{document}', content, re.DOTALL)
        if not doc_match:
            self.write("✗ Could not find document body\n", "red")
            return slides

        document_content = doc_match.group(1)

        # Extract preamble information
        preamble_info = self.extract_preamble_info(content)

        # Enhanced frame detection with multiple patterns
        frame_patterns = [
            r'\\begin{frame}(?:\[[^\]]*\])?(?:{([^}]*)})?(.*?)\\end{frame}',
            r'\\frame(?:\[[^\]]*\])?(?:{([^}]*)})?{(.*?)}'
        ]

        all_frames = []
        for pattern in frame_patterns:
            frames = re.finditer(pattern, document_content, re.DOTALL)
            all_frames.extend(frames)

        if not all_frames:
            self.write("✗ No frames found in document\n", "red")
            return slides

        self.write(f"Found {len(all_frames)} frames in document\n", "green")

        for i, frame_match in enumerate(all_frames):
            try:
                title = frame_match.group(1) if frame_match.group(1) else ""
                frame_content = frame_match.group(2) if frame_match.group(2) else ""

                # Skip title page frames but preserve their content
                if '\\titlepage' in frame_content or '\\maketitle' in frame_content:
                    self.write(f"Processing title frame {i+1}\n", "yellow")
                    # Extract title page content - use the correct method name
                    try:
                        title_slide = self.process_title_frame(frame_content, preamble_info)
                        if title_slide:
                            slides.append(title_slide)
                    except AttributeError:
                        # If process_title_frame doesn't exist, create a basic title slide
                        self.write(f"Warning: Could not process title frame {i+1}\n", "yellow")
                        title_slide = {
                            'title': 'Title Page',
                            'media': '',
                            'content': [
                                f"\\title{{{preamble_info.get('title', '')}}}",
                                f"\\author{{{preamble_info.get('author', '')}}}",
                                f"\\date{{{preamble_info.get('date', '')}}}",
                                f"\\institute{{{preamble_info.get('institute', '')}}}",
                                "\\titlepage"
                            ],
                            'notes': [],
                            'preamble_info': preamble_info
                        }
                        slides.append(title_slide)
                    continue

                # Extract frametitle if no title in frame declaration
                if not title:
                    ft_match = re.search(r'\\frametitle{([^}]*)}', frame_content)
                    if ft_match:
                        title = ft_match.group(1)

                # Extract framesubtitle
                subtitle = ""
                subtitle_match = re.search(r'\\framesubtitle{([^}]*)}', frame_content)
                if subtitle_match:
                    subtitle = subtitle_match.group(1)

                # Clean title and subtitle
                if title:
                    title = self.clean_latex_content(title)
                else:
                    title = f"Slide {i+1}"

                if subtitle:
                    title += f" - {self.clean_latex_content(subtitle)}"

                # Extract media including TikZ diagrams
                media = self.extract_enhanced_media_from_frame(frame_content)

                # Extract content with enhanced LaTeX support
                content_items = self.extract_enhanced_content_from_frame(frame_content)

                # Extract notes with proper formatting
                notes = self.extract_enhanced_notes_from_frame(frame_content)

                slide_data = {
                    'title': title,
                    'media': media,
                    'content': content_items,
                    'notes': notes,
                    'preamble_info': preamble_info
                }

                slides.append(slide_data)
                self.write(f"✓ Processed slide: {title}\n", "green")

            except Exception as e:
                self.write(f"✗ Error processing frame {i+1}: {str(e)}\n", "red")
                continue

        return slides

    def extract_preamble_info(self, content: str) -> dict:
        """Extract preamble information including themes and packages"""
        preamble_info = {
            'theme': '',
            'packages': [],
            'title': '',
            'author': '',
            'date': '',
            'institute': ''
        }

        import re

        # Extract theme
        theme_match = re.search(r'\\usetheme{([^}]*)}', content)
        if theme_match:
            preamble_info['theme'] = theme_match.group(1)

        # Extract packages
        package_matches = re.findall(r'\\usepackage(?:\[[^\]]*\])?{([^}]*)}', content)
        preamble_info['packages'] = package_matches

        # Extract document info
        title_match = re.search(r'\\title{([^}]*)}', content)
        if title_match:
            preamble_info['title'] = self.clean_latex_content(title_match.group(1))

        author_match = re.search(r'\\author{([^}]*)}', content)
        if author_match:
            preamble_info['author'] = self.clean_latex_content(author_match.group(1))

        date_match = re.search(r'\\date{([^}]*)}', content)
        if date_match:
            preamble_info['date'] = self.clean_latex_content(date_match.group(1))

        institute_match = re.search(r'\\institute{([^}]*)}', content)
        if institute_match:
            preamble_info['institute'] = self.clean_latex_content(institute_match.group(1))

        return preamble_info

    def process_title_frame(self, frame_content: str, preamble_info: dict) -> dict:
        """Process title page frames with better error handling"""
        try:
            title_slide = {
                'title': 'Title Page',
                'media': '',
                'content': [],
                'notes': [],
                'preamble_info': preamble_info  # Add this to preserve preamble info
            }

            # Add title information to content with safe dictionary access
            if preamble_info.get('title'):
                title_slide['content'].append(f"\\title{{{preamble_info['title']}}}")

            if preamble_info.get('author'):
                title_slide['content'].append(f"\\author{{{preamble_info['author']}}}")

            if preamble_info.get('date'):
                title_slide['content'].append(f"\\date{{{preamble_info['date']}}}")

            if preamble_info.get('institute'):
                title_slide['content'].append(f"\\institute{{{preamble_info['institute']}}}")

            # Add title page command
            title_slide['content'].append("\\titlepage")

            return title_slide

        except Exception as e:
            # Log the error but return a basic title slide
            print(f"Warning: Error in process_title_frame: {str(e)}")
            # Return minimal title slide
            return {
                'title': 'Title Page',
                'media': '',
                'content': ["\\titlepage"],
                'notes': [],
                'preamble_info': preamble_info
            }

    def extract_enhanced_media_from_frame(self, frame_content: str) -> str:
        """Extract media including TikZ diagrams and graphics"""
        import re

        media_components = []

        # Check for graphics
        graphics_matches = re.finditer(r'\\includegraphics(?:\[[^\]]*\])?{([^}]*)}', frame_content)
        for match in graphics_matches:
            media_path = match.group(1)
            media_components.append(f"\\file {media_path}")

        # Check for TikZ environments
        tikz_matches = re.finditer(r'\\begin{tikzpicture}(.*?)\\end{tikzpicture}', frame_content, re.DOTALL)
        for i, match in enumerate(tikz_matches):
            tikz_content = match.group(0).strip()
            # Save TikZ code as a separate file or include it directly
            filename = f"tikz_diagram_{i+1}.tex"
            media_components.append(f"% TikZ Diagram {i+1}:\n{tikz_content}")

        # Check for movies/videos
        movie_match = re.search(r'\\movie(?:\[[^\]]*\])?{[^}]*}{([^}]*)}', frame_content)
        if movie_match:
            media_path = movie_match.group(1)
            media_components.append(f"\\play {media_path}")

        return "\n".join(media_components) if media_components else ""

    def extract_enhanced_content_from_frame(self, frame_content: str) -> list:
        """Extract content with enhanced LaTeX feature support"""
        import re

        content_items = []

        # Remove notes and media commands to isolate main content
        clean_content = re.sub(r'\\note{.*?}', '', frame_content, flags=re.DOTALL)
        clean_content = re.sub(r'\\includegraphics[^}]*}|\\movie[^}]*}|\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', '', clean_content, flags=re.DOTALL)

        # Extract blocks (block, example, alertblock)
        block_patterns = [
            (r'\\begin\{block\}\{([^}]*)\}(.*?)\\end\{block\}', 'block'),
            (r'\\begin\{example\}\{([^}]*)\}(.*?)\\end\{example\}', 'example'),
            (r'\\begin\{alertblock\}\{([^}]*)\}(.*?)\\end\{alertblock\}', 'alertblock'),
            (r'\\begin\{columns\}(.*?)\\end\{columns\}', 'columns')
        ]

        for pattern, block_type in block_patterns:
            blocks = re.finditer(pattern, clean_content, re.DOTALL)
            for block in blocks:
                if block_type == 'columns':
                    # Process columns environment
                    columns_content = self.process_columns_environment(block.group(1))
                    content_items.extend(columns_content)
                else:
                    title = self.clean_latex_content(block.group(1))
                    content = self.clean_latex_content(block.group(2))
                    content_items.append(f"\\begin{{{block_type}}}{{{title}}}")
                    content_items.extend(self.process_block_content(content))
                    content_items.append(f"\\end{{{block_type}}}")

        # Extract items from itemize and enumerate environments
        list_environments = ['itemize', 'enumerate']
        for env in list_environments:
            env_matches = re.finditer(r'\\begin\{' + env + r'\}(.*?)\\end\{' + env + r'\}', clean_content, re.DOTALL)
            for match in env_matches:
                env_content = match.group(1)
                items = re.finditer(r'\\item\s*(.*?)(?=\\item|\\end\{' + env + r'\}|$)', env_content, re.DOTALL)
                for item in items:
                    item_text = self.clean_latex_content(item.group(1).strip())
                    if item_text:
                        content_items.append(f"- {item_text}")

        # Extract standalone equations
        equation_matches = re.finditer(r'\\\[(.*?)\\\]', clean_content, re.DOTALL)
        for match in equation_matches:
            equation = match.group(1).strip()
            content_items.append(f"\\[{equation}\\]")

        # Extract inline math and other content
        paragraphs = re.split(r'\n\s*\n', clean_content)
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph and not any(env in paragraph for env in ['block', 'example', 'alertblock', 'columns', 'itemize', 'enumerate']):
                # Clean up the paragraph but preserve LaTeX commands
                paragraph = self.preserve_latex_commands(paragraph)
                if paragraph and len(paragraph) > 3:
                    content_items.append(paragraph)

        return content_items

    def extract_enhanced_notes_from_frame(self, frame_content: str) -> list:
        """Extract speaker notes with proper formatting"""
        import re

        notes = []

        # Extract note commands with [item] support
        note_matches = re.finditer(r'\\note(?:\[([^\]]*)\])?\{(.*?)\}', frame_content, re.DOTALL)
        for match in note_matches:
            note_type = match.group(1)  # [item] or None
            note_content = match.group(2).strip()

            if note_type == 'item':
                # Process itemized notes
                items = re.finditer(r'\\item\s*(.*?)(?=\\item|$)', note_content, re.DOTALL)
                for item in items:
                    item_text = self.clean_latex_content(item.group(1).strip())
                    if item_text:
                        notes.append(f"• {item_text}")
            else:
                # Regular note content
                lines = note_content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:
                        notes.append(f"• {self.clean_latex_content(line)}")

        return notes

    def process_block_content(self, content: str) -> list:
        """Process content within blocks"""
        import re

        lines = []

        # Extract items from block content
        items = re.finditer(r'\\item\s*(.*?)(?=\\item|$)', content, re.DOTALL)
        for item in items:
            item_text = self.clean_latex_content(item.group(1).strip())
            if item_text:
                lines.append(f"- {item_text}")

        # If no items found, treat as regular content
        if not lines:
            paragraphs = content.split('\n')
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if paragraph:
                    lines.append(self.preserve_latex_commands(paragraph))

        return lines

    def preserve_latex_commands(self, text: str) -> str:
        """Clean text while preserving important LaTeX commands"""
        import re

        # Preserve math environments
        text = re.sub(r'\\\[(.*?)\\\]', r'\\[\1\\]', text)
        text = re.sub(r'\\\((.*?)\\\)', r'\\(\1\\)', text)

        # Preserve text formatting commands
        text = re.sub(r'\\textbf{([^}]*)}', r'\\textbf{\1}', text)
        text = re.sub(r'\\textit{([^}]*)}', r'\\textit{\1}', text)
        text = re.sub(r'\\emph{([^}]*)}', r'\\emph{\1}', text)

        # Preserve other common commands
        text = re.sub(r'\\mathbb{([^}]*)}', r'\\mathbb{\1}', text)
        text = re.sub(r'\\mathcal{([^}]*)}', r'\\mathcal{\1}', text)
        text = re.sub(r'\\bm{([^}]*)}', r'\\bm{\1}', text)

        # Clean up extra whitespace but preserve command structure
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def clean_latex_content(self, text: str) -> str:
        """Clean LaTeX content while preserving structure"""
        import re

        if not text:
            return ""

        # Remove comments
        text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)

        # Replace common LaTeX commands with plain text equivalents
        replacements = [
            (r'\\textcolor{[^}]*}{([^}]*)}', r'\1'),
            (r'~', ' '),
            (r'\\&', '&'),
            (r'\\%', '%'),
            (r'\\#', '#'),
            (r'\\_', '_'),
            (r'\\ldots', '...'),
            (r'\\quad', '    '),
            (r'\\qquad', '        '),
        ]

        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def load_tex_file(self) -> None:
        """Load and convert a Beamer .tex file to IDE format using simple converter"""
        tex_file = filedialog.askopenfilename(
            filetypes=[("TeX files", "*.tex"), ("All files", "*.*")],
            title="Select Beamer TeX File to Load"
        )

        if not tex_file:
            return

        try:
            # Convert TeX to simple text format
            text_file = convert_beamer_tex_to_simple_text(tex_file)

            if not text_file:
                messagebox.showerror("Error", "Failed to convert TeX file")
                return

            # Now load the converted text file using the existing load_file method
            self.load_file(text_file)

            self.write(f"✓ Successfully loaded and converted: {os.path.basename(tex_file)}\n", "green")

            # Ask if user wants to generate PDF immediately
            if messagebox.askyesno("Success",
                                 "TeX file converted successfully!\n\n"
                                 "Would you like to generate PDF now?"):
                self.generate_pdf()

        except Exception as e:
            error_msg = f"Error loading TeX file:\n{str(e)}"
            self.write(f"✗ {error_msg}\n", "red")
            messagebox.showerror("Error", error_msg)

    def extract_preamble_info_safe(self, content: str) -> dict:
        """Extract preamble information with safer regex patterns"""
        import re

        preamble_info = {
            'theme': '',
            'packages': [],
            'title': '',
            'author': '',
            'date': '',
            'institute': ''
        }

        try:
            # Extract theme with safer pattern
            theme_match = re.search(r'\\usetheme\s*\{([^}]+)\}', content)
            if theme_match:
                preamble_info['theme'] = theme_match.group(1).strip()

            # Extract packages with safer pattern
            package_matches = re.findall(r'\\usepackage\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}', content)
            if package_matches:
                preamble_info['packages'] = [pkg.strip() for pkg in package_matches]

            # Extract document info with safer patterns
            title_match = re.search(r'\\title\s*\{((?:[^{}]|\{[^{}]*\})*)\}', content, re.DOTALL)
            if title_match:
                preamble_info['title'] = self.clean_latex_content_safe(title_match.group(1))

            author_match = re.search(r'\\author\s*\{((?:[^{}]|\{[^{}]*\})*)\}', content, re.DOTALL)
            if author_match:
                preamble_info['author'] = self.clean_latex_content_safe(author_match.group(1))

            date_match = re.search(r'\\date\s*\{((?:[^{}]|\{[^{}]*\})*)\}', content, re.DOTALL)
            if date_match:
                preamble_info['date'] = self.clean_latex_content_safe(date_match.group(1))

            institute_match = re.search(r'\\institute\s*\{((?:[^{}]|\{[^{}]*\})*)\}', content, re.DOTALL)
            if institute_match:
                preamble_info['institute'] = self.clean_latex_content_safe(institute_match.group(1))

        except Exception as e:
            self.write(f"Warning: Error extracting preamble info: {str(e)}\n", "yellow")

        return preamble_info

    def enhanced_extract_slides_from_tex_safe(self, content: str) -> list:
        """Enhanced slide extraction with comprehensive LaTeX feature support and better error handling"""
        slides = []
        import re

        try:
            # First isolate the document body
            doc_match = re.search(r'\\begin{document}(.*?)\\end{document}', content, re.DOTALL)
            if not doc_match:
                self.write("✗ Could not find document body\n", "red")
                return slides

            document_content = doc_match.group(1)

            # Extract preamble information
            preamble_info = self.extract_preamble_info_safe(content)

            # Enhanced frame detection with multiple patterns and better handling
            # Pattern 1: \begin{frame}...\end{frame}
            frame_pattern = r'\\begin\{frame\}(.*?)\\end\{frame\}'

            all_frames = []
            try:
                frames = re.finditer(frame_pattern, document_content, re.DOTALL)
                all_frames = list(frames)
            except Exception as e:
                self.write(f"Warning: Error finding frames with pattern 1: {str(e)}\n", "yellow")
                # Try alternative pattern
                try:
                    # Pattern 2: \frame{...}
                    frame_pattern2 = r'\\frame\{((?:[^{}]|\{[^{}]*\})*)\}'
                    frames = re.finditer(frame_pattern2, document_content, re.DOTALL)
                    all_frames = list(frames)
                except Exception as e2:
                    self.write(f"Warning: Error finding frames with pattern 2: {str(e2)}\n", "yellow")

            if not all_frames:
                self.write("✗ No frames found in document\n", "red")
                return slides

            self.write(f"Found {len(all_frames)} frames in document\n", "green")

            for i, frame_match in enumerate(all_frames):
                try:
                    frame_content = frame_match.group(1) if frame_match.groups() else frame_match.group(0)

                    # Skip title page frames but preserve their content
                    if '\\titlepage' in frame_content or '\\maketitle' in frame_content:
                        self.write(f"Processing title frame {i+1}\n", "yellow")
                        # Extract title page content - USE THE ORIGINAL METHOD
                        title_slide = self.process_title_frame(frame_content, preamble_info)
                        if title_slide:
                            slides.append(title_slide)
                        continue

                    # Extract title
                    title = ""
                    # Try frametitle first
                    ft_match = re.search(r'\\frametitle\s*\{((?:[^{}]|\{[^{}]*\})*)\}', frame_content, re.DOTALL)
                    if ft_match:
                        title = self.clean_latex_content_safe(ft_match.group(1))

                    # If no frametitle, check for frame title in options
                    if not title:
                        # Check for frame with title option: \begin{frame}{Title}
                        title_match = re.match(r'^\s*\{((?:[^{}]|\{[^{}]*\})*)\}', frame_content.strip())
                        if title_match:
                            title = self.clean_latex_content_safe(title_match.group(1))

                    if not title:
                        title = f"Slide {i+1}"

                    # Extract framesubtitle
                    subtitle = ""
                    subtitle_match = re.search(r'\\framesubtitle\s*\{((?:[^{}]|\{[^{}]*\})*)\}', frame_content, re.DOTALL)
                    if subtitle_match:
                        subtitle = self.clean_latex_content_safe(subtitle_match.group(1))
                        title += f" - {subtitle}"

                    # Extract media including TikZ diagrams - USE THE ORIGINAL METHOD
                    media = self.extract_enhanced_media_from_frame(frame_content)

                    # Extract content with enhanced LaTeX support - USE THE ORIGINAL METHOD
                    content_items = self.extract_enhanced_content_from_frame(frame_content)

                    # Extract notes with proper formatting - USE THE ORIGINAL METHOD
                    notes = self.extract_enhanced_notes_from_frame(frame_content)

                    slide_data = {
                        'title': title,
                        'media': media,
                        'content': content_items,
                        'notes': notes,
                        'preamble_info': preamble_info
                    }

                    slides.append(slide_data)
                    self.write(f"✓ Processed slide: {title}\n", "green")

                except Exception as e:
                    self.write(f"✗ Error processing frame {i+1}: {str(e)}\n", "red")
                    # Add a placeholder slide to maintain slide count
                    slides.append({
                        'title': f"Slide {i+1} (Error)",
                        'media': '',
                        'content': [f"% Error processing this slide: {str(e)}"],
                        'notes': [],
                        'preamble_info': preamble_info
                    })
                    continue

            return slides

        except Exception as e:
            self.write(f"✗ Error in slide extraction: {str(e)}\n", "red")
            return []

    def clean_latex_content_safe(self, text: str) -> str:
        """Clean LaTeX content while preserving structure with safer regex handling"""
        import re

        if not text:
            return ""

        try:
            # Remove comments first
            text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)

            # Replace common LaTeX commands with plain text equivalents
            # Use raw strings and be careful with backslashes
            replacements = [
                (r'\\textcolor\{[^}]*\}\{([^}]*)\}', r'\1'),  # Remove textcolor
                (r'~', ' '),  # Replace non-breaking space
                (r'\\&', '&'),
                (r'\\%', '%'),
                (r'\\#', '#'),
                (r'\\_', '_'),
                (r'\\ldots', '...'),
                (r'\\quad', '    '),
                (r'\\qquad', '        '),
            ]

            for pattern, replacement in replacements:
                try:
                    text = re.sub(pattern, replacement, text)
                except:
                    pass  # Skip if pattern causes issues

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

            return text

        except Exception as e:
            self.write(f"Warning: Error cleaning LaTeX content: {str(e)}\n", "yellow")
            # Return the original text if cleaning fails
            return text

    def extract_enhanced_media_from_frame_safe(self, frame_content: str) -> str:
        """Extract media including TikZ diagrams with complete code preservation"""
        import re

        media_components = []

        try:
            # Check for TikZ environments - preserve COMPLETE code
            tikz_pattern = r'\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}'
            tikz_matches = re.finditer(tikz_pattern, frame_content, re.DOTALL)

            for i, match in enumerate(tikz_matches):
                tikz_content = match.group(0).strip()  # Get the ENTIRE TikZ code
                # Preserve the complete TikZ code as-is
                media_components.append(f"% TikZ Diagram:\n{tikz_content}")

            # Check for graphics
            graphics_pattern = r'\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}'
            graphics_matches = re.finditer(graphics_pattern, frame_content)

            for match in graphics_matches:
                media_path = match.group(1).strip()
                media_components.append(f"\\file {media_path}")

            # Check for movies/videos
            movie_pattern = r'\\movie\s*(?:\[[^\]]*\])?\s*\{[^}]*\}\s*\{([^}]+)\}'
            movie_match = re.search(movie_pattern, frame_content)
            if movie_match:
                media_path = movie_match.group(1).strip()
                media_components.append(f"\\play {media_path}")

            # Join with double newlines to ensure proper separation
            return "\n\n".join(media_components) if media_components else ""

        except Exception as e:
            self.write(f"Warning: Error extracting media: {str(e)}\n", "yellow")
            return ""

    def extract_enhanced_content_from_frame_safe(self, frame_content: str) -> list:
        """Extract content with enhanced LaTeX feature support and safer patterns"""
        import re

        content_items = []

        try:
            # Remove notes and media commands to isolate main content
            # Use safer patterns that don't cause regex issues
            clean_content = frame_content

            # Remove note commands
            clean_content = re.sub(r'\\note\s*\{[^}]*\}', '', clean_content, flags=re.DOTALL)

            # Remove graphics commands
            clean_content = re.sub(r'\\includegraphics[^}]*\}', '', clean_content)

            # Remove movie commands
            clean_content = re.sub(r'\\movie[^}]*\}', '', clean_content)


            # Extract blocks with safer patterns
            block_patterns = [
                (r'\\begin\{block\}\s*\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{block\}', 'block'),
                (r'\\begin\{example\}\s*\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{example\}', 'example'),
                (r'\\begin\{alertblock\}\s*\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{alertblock\}', 'alertblock'),
                (r'\\begin\{columns\}(.*?)\\end\{columns\}', 'columns')
            ]

            for pattern, block_type in block_patterns:
                try:
                    blocks = re.finditer(pattern, clean_content, re.DOTALL)
                    for block in blocks:
                        if block_type == 'columns':
                            # Process columns environment
                            columns_content = self.process_columns_environment_safe(block.group(1))
                            content_items.extend(columns_content)
                        else:
                            title = self.clean_latex_content_safe(block.group(1))
                            content = self.clean_latex_content_safe(block.group(2))
                            content_items.append(f"\\begin{{{block_type}}}{{{title}}}")
                            content_items.extend(self.process_block_content_safe(content))
                            content_items.append(f"\\end{{{block_type}}}")
                except:
                    continue  # Skip if pattern causes issues

            # Extract items from itemize and enumerate environments with safer patterns
            list_environments = ['itemize', 'enumerate']
            for env in list_environments:
                try:
                    env_pattern = f'\\\\begin\\{{{env}\\}}(.*?)\\\\end\\{{{env}\\}}'
                    env_matches = re.finditer(env_pattern, clean_content, re.DOTALL)
                    for match in env_matches:
                        env_content = match.group(1)
                        item_pattern = r'\\item\s*(.*?)(?=\\item|\\end\{' + env + r'\}|$)'
                        items = re.finditer(item_pattern, env_content, re.DOTALL)
                        for item in items:
                            item_text = self.clean_latex_content_safe(item.group(1).strip())
                            if item_text:
                                content_items.append(f"- {item_text}")
                except:
                    continue

            # Extract standalone equations with safer pattern
            try:
                equation_pattern = r'\\\[(.*?)\\\]'
                equation_matches = re.finditer(equation_pattern, clean_content, re.DOTALL)
                for match in equation_matches:
                    equation = match.group(1).strip()
                    content_items.append(f"\\[{equation}\\]")
            except:
                pass

            # Extract inline math and other content
            paragraphs = re.split(r'\n\s*\n', clean_content)
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if paragraph and not any(env in paragraph for env in ['block', 'example', 'alertblock', 'columns', 'itemize', 'enumerate']):
                    # Clean up the paragraph but preserve LaTeX commands
                    paragraph = self.preserve_latex_commands_safe(paragraph)
                    if paragraph and len(paragraph) > 3:
                        content_items.append(paragraph)

            return content_items

        except Exception as e:
            self.write(f"Warning: Error extracting content: {str(e)}\n", "yellow")
            return ["% Error extracting content from this slide"]

    def preserve_latex_commands_safe(self, text: str) -> str:
        """Clean text while preserving important LaTeX commands with safer patterns"""
        import re

        try:
            # Preserve math environments with safer patterns
            text = re.sub(r'\\\[(.*?)\\\]', r'\\[\1\\]', text, flags=re.DOTALL)
            text = re.sub(r'\\\((.*?)\\\)', r'\\(\1\\)', text, flags=re.DOTALL)

            # Preserve text formatting commands with safer patterns
            commands_to_preserve = ['textbf', 'textit', 'emph', 'mathbb', 'mathcal', 'bm']
            for cmd in commands_to_preserve:
                pattern = f'\\\\{cmd}\\{{((?:[^{{}}]|\\{{[^{{}}]*\\}})*)\\}}'
                try:
                    text = re.sub(pattern, f'\\\\{cmd}{{\\1}}', text, flags=re.DOTALL)
                except:
                    pass  # Skip if pattern causes issues

            # Clean up extra whitespace but preserve command structure
            text = re.sub(r'\s+', ' ', text)

            return text.strip()

        except Exception as e:
            self.write(f"Warning: Error preserving LaTeX commands: {str(e)}\n", "yellow")
            return text

    def extract_media_from_frame(self, frame_content: str) -> str:
        """Extract media references from frame content"""
        import re

        media = ""

        # Check for graphics
        graphics_match = re.search(r'\\includegraphics(?:\[[^\]]*\])?{([^}]*)}', frame_content)
        if graphics_match:
            media_path = graphics_match.group(1)
            media = f"\\file {media_path}"

        # Check for movies/videos
        movie_match = re.search(r'\\movie(?:\[[^\]]*\])?{[^}]*}{([^}]*)}', frame_content)
        if movie_match:
            media_path = movie_match.group(1)
            media = f"\\play {media_path}"

        return media

    def extract_content_from_frame(self, frame_content: str) -> list:
        """Extract content items from frame - treat as plain text unless explicit \item"""
        import re

        content_items = []

        # Remove media commands and notes to isolate main content
        clean_content = re.sub(r'\\includegraphics[^}]*}|\\movie[^}]*}|\\note{[^}]*}', '', frame_content)

        # Only extract explicit \item commands with bullets
        # Look for \item commands anywhere in the content
        items = re.finditer(r'\\item\s*(.*?)(?=\\item|\\end{itemize}|\\end{enumerate}|$)',
                           clean_content, re.DOTALL)

        for item in items:
            item_text = self.clean_latex_content(item.group(1).strip())
            if item_text:
                content_items.append(f"- {item_text}")

        # If no explicit items found, extract the main content as plain text blocks
        if not content_items:
            # Remove all LaTeX environments to get plain text
            plain_content = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', clean_content, flags=re.DOTALL)
            plain_content = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', plain_content)  # Remove other LaTeX commands

            # Split into paragraphs and clean
            paragraphs = re.split(r'\n\s*\n', plain_content)
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                # Clean up the paragraph
                paragraph = re.sub(r'\s+', ' ', paragraph)  # Normalize whitespace
                paragraph = self.clean_latex_content(paragraph)

                if paragraph and len(paragraph) > 3:  # Minimum length
                    content_items.append(paragraph)  # No bullet for plain text

        return content_items

    def extract_notes_from_frame(self, frame_content: str) -> list:
        """Extract speaker notes from frame"""
        import re

        notes = []

        note_match = re.search(r'\\note{(.*?)}', frame_content, re.DOTALL)
        if note_match:
            note_content = note_match.group(1)
            # Extract items from note content
            note_items = re.finditer(r'\\item\s*(.*?)(?=\\item|$)', note_content, re.DOTALL)
            for item in note_items:
                note_text = self.clean_latex_content(item.group(1).strip())
                if note_text:
                    notes.append(f"• {note_text}")

        return notes

    def overwrite_tex_and_generate_pdf(self) -> None:
        """Convert current presentation back to TeX and generate PDF"""
        if not self.current_file:
            messagebox.showwarning("Warning", "Please save your presentation first!")
            return

        try:
            # Save current state
            self.save_current_slide()
            self.save_file()

            # Get corresponding .tex filename
            base_name = os.path.splitext(self.current_file)[0]
            if base_name.endswith('_converted'):
                base_name = base_name[:-10]  # Remove _converted suffix

            tex_file = base_name + '.tex'

            # Ask for confirmation before overwriting
            if os.path.exists(tex_file):
                if not messagebox.askyesno("Confirm Overwrite",
                                         f"The file '{os.path.basename(tex_file)}' already exists.\n\n"
                                         "Do you want to overwrite it?"):
                    return

            self.write(f"Converting to TeX: {tex_file}\n", "white")

            # Convert to TeX
            self.convert_to_tex()

            if not os.path.exists(tex_file):
                self.write("✗ TeX file was not created successfully\n", "red")
                return

            self.write("✓ TeX file created successfully\n", "green")

            # Generate PDF
            self.write("Generating PDF...\n", "white")
            self.generate_pdf()

            # Preview PDF
            pdf_file = base_name + '.pdf'
            if os.path.exists(pdf_file):
                if messagebox.askyesno("Success",
                                     "PDF generated successfully!\n\n"
                                     "Would you like to view it now?"):
                    self.preview_pdf(pdf_file)

        except Exception as e:
            error_msg = f"Error in TeX conversion and PDF generation:\n{str(e)}"
            self.write(f"✗ {error_msg}\n", "red")
            messagebox.showerror("Error", error_msg)

#-------------------------------------------------------------------------------------
    def setup_spellchecking(self):
        """Initialize spell checking with always-on enforcement and automatic dictionary handling"""
        try:
            from spellchecker import SpellChecker

            # ENFORCE ALWAYS ON - Remove any disable capability
            self.spell_checking_enabled = True  # Force always enabled

            # Initialize spell checker with default language
            self.spell_checker = SpellChecker()

            # Language and dialect support
            self.available_languages = self.load_available_languages()
            self.current_language = 'en'
            self.current_language_name = 'English (US)'

            # Spell check settings
            self.case_sensitive = False
            self.ignore_numbers = True

            # Use unique tag names
            self.spell_tags = {
                'misspelled': 'spell_misspelled',
                'misspelled_highlight': 'spell_highlight',
            }

            # VERIFY AND PROCURE DICTIONARIES
            self.verify_dictionaries_available()

            for editor in [self.content_editor._textbox, self.notes_editor._textbox]:
                # Configure spell checking tags
                editor.tag_configure(self.spell_tags['misspelled'],
                                   underline=True, underlinefg="red")
                editor.tag_configure(self.spell_tags['misspelled_highlight'],
                                   background="#2F3542")

                # Bind spellcheck events
                editor.bind("<Button-1>", self.on_text_click_spellcheck)
                editor.bind("<Button-3>", self.on_right_click_spellcheck)
                editor.bind("<KeyRelease>", self.delayed_spell_check, add='+')

            print("✓ Spell checking ENABLED (always on)")
            self.write("✓ Spell checking initialized with dictionary verification\n", "green")

            # Create spelling context menu and anchored tooltip
            self.create_spelling_context_menu()
            self.create_anchored_tooltip()

            # Add language selection to toolbar
            self.add_language_selection_to_toolbar()

            # Perform initial spell check
            self.after(1000, self.perform_initial_spell_check)

        except ImportError as e:
            self.spell_checking_enabled = False
            error_msg = f"✗ Spell checking disabled: pyspellchecker not installed: {e}"
            print(error_msg)
            self.write(f"{error_msg}\n", "red")
        except Exception as e:
            error_msg = f"✗ Error initializing spell checking: {str(e)}"
            print(error_msg)
            self.write(f"{error_msg}\n", "red")
            import traceback
            traceback.print_exc()

    def verify_dictionaries_available(self):
        """Verify dictionaries are available and attempt to procure missing ones"""
        try:
            print("Verifying spell check dictionaries...")

            # Test basic spell checking functionality
            test_words = ['test', 'spelling', 'verification']
            known_words = self.spell_checker.known(test_words)

            if len(known_words) == len(test_words):
                print("✓ Default English dictionary is available")
                self.write("✓ Default English dictionary loaded\n", "green")
            else:
                print("! Default dictionary may be incomplete")
                self.write("! Default dictionary may need enhancement\n", "yellow")

            # Check if we can load additional languages
            self.attempt_dictionary_procurement()

        except Exception as e:
            error_msg = f"Error verifying dictionaries: {str(e)}"
            print(f"✗ {error_msg}")
            self.write(f"✗ {error_msg}\n", "red")

    def attempt_dictionary_procurement(self):
        """Attempt to procure and load additional dictionaries - FIXED VERSION"""
        try:
            print("Checking for additional dictionaries...")

            # List of languages to attempt loading
            target_languages = ['en_GB', 'es', 'fr', 'de']

            for lang_code in target_languages:
                try:
                    # Import here to avoid scope issues
                    from spellchecker import SpellChecker

                    # Try to create a spell checker with this language
                    test_speller = SpellChecker(language=lang_code)
                    # Test with a simple word
                    test_word = 'test'
                    if test_speller.correction(test_word) == test_word:
                        lang_name = self.get_language_name(lang_code)
                        print(f"✓ Dictionary available: {lang_name} ({lang_code})")
                        self.write(f"✓ Loaded dictionary: {lang_name}\n", "green")
                except Exception as e:
                    print(f"✗ Dictionary not available: {lang_code} - {e}")
                    continue

        except Exception as e:
            print(f"Error during dictionary procurement: {e}")

    def get_language_name(self, lang_code):
        """Get display name for language code"""
        language_names = {
            'en': 'English (US)',
            'en_GB': 'English (UK)',
            'en_CA': 'English (Canada)',
            'en_AU': 'English (Australia)',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese'
        }
        return language_names.get(lang_code, lang_code)

    def load_available_languages(self):
        """Load available languages with enhanced error handling"""
        languages = {
            'English (US)': 'en',
            'English (UK)': 'en_GB',
            'English (Canada)': 'en_CA',
            'English (Australia)': 'en_AU',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Dutch': 'nl',
            'Russian': 'ru',
            'Chinese (Simplified)': 'zh',
            'Japanese': 'ja',
            'Korean': 'ko',
            'Arabic': 'ar',
            'Hindi': 'hi'
        }

        available_languages = {'English (US)': 'en'}  # Always available

        try:
            from spellchecker import SpellChecker

            print("Loading available dictionaries...")

            for lang_name, lang_code in languages.items():
                try:
                    # Try to create a spell checker with this language
                    test_speller = SpellChecker(language=lang_code)
                    # Test functionality
                    test_word = 'test'
                    correction = test_speller.correction(test_word)

                    if correction:  # If we get any correction, dictionary works
                        available_languages[lang_name] = lang_code
                        print(f"✓ Loaded dictionary: {lang_name}")
                        self.write(f"✓ Dictionary available: {lang_name}\n", "green")
                    else:
                        print(f"✗ Dictionary empty or invalid: {lang_name}")

                except Exception as e:
                    print(f"✗ Dictionary not available: {lang_name} - {e}")
                    continue

        except Exception as e:
            error_msg = f"Error loading languages: {e}"
            print(error_msg)
            self.write(f"✗ {error_msg}\n", "red")

        return available_languages

    def on_text_click_spellcheck(self, event):
        """Handle left-click on text for spell checking - RESTORED"""
        if not self.spell_checking_enabled:
            return

        widget = event.widget
        index = widget.index(f"@{event.x},{event.y}")

        # Check if we're clicking on a misspelled word
        if self.spell_tags['misspelled'] in widget.tag_names(index):
            # Get the word under cursor
            word_start = widget.index(f"{index} wordstart")
            word_end = widget.index(f"{index} wordend")
            word = widget.get(word_start, word_end)

            # Remove any previous highlighting
            widget.tag_remove(self.spell_tags['misspelled_highlight'], "1.0", "end")

            # Highlight the clicked word
            widget.tag_add(self.spell_tags['misspelled_highlight'], word_start, word_end)

            # Show anchored tooltip (popup window)
            self.show_anchored_tooltip(widget, word, word_start, word_end, event.x_root, event.y_root)

            # Prevent event from propagating to other handlers
            return "break"
        else:
            # Clicked on normal text, hide tooltip
            self.hide_anchored_tooltip()

    def on_right_click_spellcheck(self, event):
        """Handle right-click on text for spell checking (alternative method)"""
        if not self.spell_checking_enabled:
            return

        widget = event.widget
        index = widget.index(f"@{event.x},{event.y}")

        # Check if right-click is on a misspelled word
        if self.spell_tags['misspelled'] in widget.tag_names(index):
            # Get the word and its position
            word_start = widget.index(f"{index} wordstart")
            word_end = widget.index(f"{index} wordend")
            word = widget.get(word_start, word_end)

            # Store current word info
            self.current_spellcheck_word = {
                'widget': widget,
                'word': word,
                'start': word_start,
                'end': word_end
            }

            # Highlight the word temporarily
            widget.tag_remove(self.spell_tags['misspelled_highlight'], "1.0", "end")
            widget.tag_add(self.spell_tags['misspelled_highlight'], word_start, word_end)

            # Show context menu instead of tooltip
            self.show_spelling_suggestions_menu(event.x_root, event.y_root, word)

            return "break"

    def create_anchored_tooltip(self):
        """Create the popup tooltip window for spell suggestions - RESTORED"""
        self.anchored_tooltip = tk.Toplevel(self)
        self.anchored_tooltip.wm_overrideredirect(True)
        self.anchored_tooltip.configure(bg='#ffffe0', relief='solid', borderwidth=2)
        self.anchored_tooltip.attributes('-topmost', True)

        # Title frame
        self.title_frame = tk.Frame(self.anchored_tooltip, bg='#ffffe0')
        self.title_frame.pack(fill='x', padx=8, pady=(8, 4))

        self.title_label = tk.Label(
            self.title_frame,
            text="Spelling Suggestions:",
            bg='#ffffe0',
            fg='#333333',
            font=("Arial", 9, "bold"),
            justify='left'
        )
        self.title_label.pack(anchor='w')

        # Suggestions frame
        self.suggestions_frame = tk.Frame(self.anchored_tooltip, bg='#ffffe0')
        self.suggestions_frame.pack(fill='x', padx=8, pady=4)

        # Control frame
        self.control_frame = tk.Frame(self.anchored_tooltip, bg='#ffffe0')
        self.control_frame.pack(fill='x', padx=8, pady=(4, 8))

        self.control_label = tk.Label(
            self.control_frame,
            text="↑↓: Navigate • Enter: Select • Esc: Close",
            bg='#ffffe0',
            fg='#666666',
            font=("Arial", 8),
            justify='left'
        )
        self.control_label.pack(anchor='w')

        # Initially hide the tooltip
        self.anchored_tooltip.withdraw()

        # Keyboard navigation
        self.anchored_tooltip.bind("<Escape>", lambda e: self.hide_anchored_tooltip())
        self.anchored_tooltip.bind("<KeyPress>", self.on_tooltip_keypress)

        # Track selection
        self.current_suggestions = []
        self.selected_index = 0
        self.anchored_word_info = None

    def show_anchored_tooltip(self, widget, word, word_start, word_end, x_root, y_root):
        """Show the popup tooltip with spelling suggestions - with enhanced error handling"""
        try:
            # Get spelling suggestions
            try:
                suggestions = list(self.spell_checker.candidates(word))[:8]
                if not suggestions:
                    # Fallback to correction
                    correction = self.spell_checker.correction(word)
                    if correction and correction != word:
                        suggestions = [correction]
            except Exception as e:
                error_msg = f"Error getting suggestions: {str(e)}"
                print(f"✗ {error_msg}")
                self.write(f"✗ {error_msg}\n", "red")
                suggestions = []

            if not suggestions:
                print(f"No suggestions available for word: '{word}'")
                return

            # Store current word info
            self.anchored_word_info = {
                'widget': widget,
                'word_start': word_start,
                'word_end': word_end,
                'original_word': word
            }
            self.current_suggestions = suggestions
            self.selected_index = 0

            # Update tooltip content
            self.title_label.configure(text=f"Suggestions for '{word}':")

            # Clear previous suggestions
            for child in self.suggestions_frame.winfo_children():
                child.destroy()

            # Create suggestion buttons
            self.suggestion_labels = []
            for i, suggestion in enumerate(suggestions):
                try:
                    suggestion_label = tk.Label(
                        self.suggestions_frame,
                        text=suggestion,
                        bg='#ffffe0',
                        fg='#0066cc',
                        font=("Arial", 9),
                        cursor="hand2",
                        justify='left',
                        relief='flat',
                        padx=4,
                        pady=2
                    )
                    suggestion_label.pack(fill='x', anchor='w', pady=1)

                    # Bind click event
                    suggestion_label.bind("<Button-1>",
                        lambda e, idx=i: self.select_suggestion(idx))

                    # Hover effects
                    suggestion_label.bind("<Enter>",
                        lambda e, idx=i: self.highlight_suggestion(idx))
                    suggestion_label.bind("<Leave>",
                        lambda e, idx=i: self.unhighlight_suggestion(idx))

                    self.suggestion_labels.append(suggestion_label)
                except Exception as e:
                    print(f"Error creating suggestion label: {e}")
                    continue

            # Position tooltip (offset from click position)
            tooltip_x = x_root + 20
            tooltip_y = y_root + 20

            # Ensure tooltip stays on screen
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()

            tooltip_width = 200
            tooltip_height = 150

            if tooltip_x + tooltip_width > screen_width:
                tooltip_x = x_root - tooltip_width - 20
            if tooltip_y + tooltip_height > screen_height:
                tooltip_y = y_root - tooltip_height - 20

            self.anchored_tooltip.wm_geometry(f"+{tooltip_x}+{tooltip_y}")

            # Show and focus tooltip
            self.anchored_tooltip.deiconify()
            self.anchored_tooltip.focus_set()

            # Highlight first suggestion
            self.highlight_suggestion(0)

            print(f"✓ Showing {len(suggestions)} suggestions for '{word}'")

        except Exception as e:
            error_msg = f"Error showing spell check tooltip: {str(e)}"
            print(f"✗ {error_msg}")
            self.write(f"✗ {error_msg}\n", "red")

    def select_suggestion(self, index):
        """Select and apply a spelling suggestion from tooltip - with error handling"""
        try:
            if (not self.anchored_word_info or
                not self.current_suggestions or
                index >= len(self.current_suggestions)):
                print("Invalid suggestion selection parameters")
                return

            suggestion = self.current_suggestions[index]
            widget = self.anchored_word_info['widget']
            word_start = self.anchored_word_info['word_start']
            word_end = self.anchored_word_info['word_end']
            original_word = self.anchored_word_info['original_word']

            # Replace the word
            widget.delete(word_start, word_end)
            widget.insert(word_start, suggestion)

            # Hide tooltip
            self.hide_anchored_tooltip()

            # Re-check spelling
            self.check_spelling()

            print(f"✓ Replaced '{original_word}' with '{suggestion}'")

        except Exception as e:
            error_msg = f"Error applying spelling suggestion: {str(e)}"
            print(f"✗ {error_msg}")
            self.write(f"✗ {error_msg}\n", "red")

    def hide_anchored_tooltip(self, event=None):
        """Hide the popup tooltip - RESTORED"""
        if hasattr(self, 'anchored_word_info') and self.anchored_word_info:
            # Remove word highlighting
            widget = self.anchored_word_info['widget']
            widget.tag_remove(self.spell_tags['misspelled_highlight'], "1.0", "end")

        if hasattr(self, 'anchored_tooltip'):
            self.anchored_tooltip.withdraw()

        self.current_suggestions = []
        self.selected_index = 0
        self.anchored_word_info = None

        # Return focus to main window
        self.focus_set()

    def on_tooltip_keypress(self, event):
        """Handle keyboard navigation in tooltip - RESTORED"""
        if not self.current_suggestions:
            return

        if event.keysym == "Escape":
            self.hide_anchored_tooltip()
        elif event.keysym == "Up":
            new_index = max(0, self.selected_index - 1)
            self.highlight_suggestion(new_index)
        elif event.keysym == "Down":
            new_index = min(len(self.current_suggestions) - 1, self.selected_index + 1)
            self.highlight_suggestion(new_index)
        elif event.keysym == "Return":
            self.select_suggestion(self.selected_index)

    def highlight_suggestion(self, index):
        """Highlight a suggestion in the tooltip - RESTORED"""
        # Unhighlight previous
        if hasattr(self, 'suggestion_labels') and self.suggestion_labels:
            for i, label in enumerate(self.suggestion_labels):
                if i == self.selected_index:
                    label.configure(bg='#ffffe0', fg='#0066cc', relief='flat')

        # Highlight new
        self.selected_index = index
        if hasattr(self, 'suggestion_labels') and self.suggestion_labels:
            if 0 <= index < len(self.suggestion_labels):
                self.suggestion_labels[index].configure(
                    bg='#4A90E2',
                    fg='white',
                    relief='sunken'
                )

    def unhighlight_suggestion(self, index):
        """Unhighlight a suggestion - RESTORED"""
        if index != self.selected_index and hasattr(self, 'suggestion_labels'):
            if 0 <= index < len(self.suggestion_labels):
                self.suggestion_labels[index].configure(
                    bg='#ffffe0',
                    fg='#0066cc',
                    relief='flat'
                )

    def add_language_selection_to_toolbar(self):
        """Add language selection dropdown to toolbar"""
        if not self.spell_checking_enabled:
            return

        # Add to existing toolbar
        toolbar = self.toolbar

        # Language selection frame
        lang_frame = ctk.CTkFrame(toolbar, fg_color="transparent")
        lang_frame.pack(side="left", padx=10)

        ctk.CTkLabel(lang_frame, text="Spell Check:").pack(side="left", padx=5)

        self.language_var = ctk.StringVar(value=self.current_language_name)
        self.language_menu = ctk.CTkOptionMenu(
            lang_frame,
            values=list(self.available_languages.keys()),
            variable=self.language_var,
            command=self.change_spellcheck_language,
            width=140
        )
        self.language_menu.pack(side="left", padx=5)

        # Add load dictionary button
        self.load_dict_button = ctk.CTkButton(
            lang_frame,
            text="Load Dictionary",
            command=self.load_custom_dictionary,
            width=120
        )
        self.load_dict_button.pack(side="left", padx=5)
        self.create_tooltip(self.load_dict_button, "Load custom dictionary file")

    def change_spellcheck_language(self, language_name):
        """Change spell checking language"""
        if not self.spell_checking_enabled:
            return

        try:
            language_code = self.available_languages[language_name]

            # Reinitialize spell checker with new language
            from spellchecker import SpellChecker
            self.spell_checker = SpellChecker(language=language_code)
            self.current_language = language_code
            self.current_language_name = language_name

            # Clear and re-check all text
            self.clear_spellcheck_highlights()
            self.check_spelling()

            self.write(f"✓ Spell check language changed to {language_name}\n", "green")

        except Exception as e:
            self.write(f"✗ Error changing language: {str(e)}\n", "red")
            # Revert to previous selection
            self.language_var.set(self.current_language_name)

    def load_custom_dictionary(self):
        """Load custom dictionary from file"""
        file_path = filedialog.askopenfilename(
            title="Select Custom Dictionary File",
            filetypes=[
                ("Dictionary files", "*.dic"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            # Read custom dictionary
            with open(file_path, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if line.strip()]

            # Add words to spell checker
            for word in words:
                self.spell_checker.word_frequency.add(word)

            # Re-check spelling
            self.check_spelling()

            self.write(f"✓ Loaded custom dictionary with {len(words)} words\n", "green")
            messagebox.showinfo("Success", f"Loaded {len(words)} words from custom dictionary")

        except Exception as e:
            error_msg = f"Error loading dictionary: {str(e)}"
            self.write(f"✗ {error_msg}\n", "red")
            messagebox.showerror("Error", error_msg)

    # Keep the existing check_spelling method that works
    def check_spelling(self, event=None):
        """Check spelling in real-time"""
        if not self.spell_checking_enabled:
            return

        # Process both editors
        for editor in [self.content_editor._textbox, self.notes_editor._textbox]:
            # Remove previous spell checking markings
            editor.tag_remove(self.spell_tags['misspelled'], "1.0", "end")

            # Get all text
            content = editor.get("1.0", "end")

            # Skip if content is too short
            if len(content.strip()) < 4:
                continue

            # Find all words longer than 3 characters
            words = re.findall(r'\b\w{4,}\b', content)

            # Get positions of all words
            word_positions = []
            for word in words:
                start_index = "1.0"
                while True:
                    start_index = editor.search(rf"\y{word}\y", start_index,
                                              stopindex="end", regexp=True)
                    if not start_index:
                        break
                    end_index = f"{start_index}+{len(word)}c"
                    word_positions.append((word, start_index, end_index))
                    start_index = end_index

            # Check each word
            for word, start, end in word_positions:
                # Skip words that are in syntax highlighting tags (LaTeX commands, etc.)
                skip_tags = ['command', 'media', 'bullet', 'url', 'bracket', 'rgb', 'textcolor']

                skip = False
                for tag in skip_tags:
                    if tag in editor.tag_names(start):
                        skip = True
                        break

                if skip:
                    continue

                # Check spelling (case insensitive unless specified)
                check_word = word if self.case_sensitive else word.lower()

                # Skip words with numbers if option is enabled
                if self.ignore_numbers and any(char.isdigit() for char in word):
                    continue

                # Actual spell check
                if not self.spell_checker.known([check_word]):
                    editor.tag_add(self.spell_tags['misspelled'], start, end)

    def clear_spellcheck_highlights(self):
        """Clear all spell check highlighting"""
        for editor in [self.content_editor._textbox, self.notes_editor._textbox]:
            for tag in self.spell_tags.values():
                editor.tag_remove(tag, "1.0", "end")


    def create_spelling_context_menu(self):
        """Create context menu for spelling suggestions"""
        self.spelling_menu = tk.Menu(self, tearoff=0)

        # Suggestions will be added dynamically
        self.suggestion_items = []

        # Static menu items
        self.spelling_menu.add_separator()
        self.spelling_menu.add_command(label="Ignore All", command=self.ignore_all_occurrences)
        self.spelling_menu.add_command(label="Add to Dictionary", command=self.add_to_dictionary)
        self.spelling_menu.add_separator()
        self.spelling_menu.add_command(label="Spell Check Settings...", command=self.show_spellcheck_settings)


    def show_spelling_suggestions_menu(self, x, y, word):
        """Show spelling suggestions in context menu"""
        # Clear previous suggestions
        for item in self.suggestion_items:
            self.spelling_menu.delete(0)
        self.suggestion_items.clear()

        # Get spelling suggestions
        try:
            suggestions = list(self.spell_checker.candidates(word))[:8]  # More suggestions
            if not suggestions:
                # Fallback to correction
                correction = self.spell_checker.correction(word)
                if correction and correction != word:
                    suggestions = [correction]
        except Exception:
            suggestions = []

        # Add suggestions to menu
        if suggestions:
            for i, suggestion in enumerate(suggestions):
                self.spelling_menu.insert_command(
                    i,
                    label=suggestion,
                    command=lambda s=suggestion: self.replace_spelling_suggestion(s)
                )
                self.suggestion_items.append(suggestion)
        else:
            self.spelling_menu.insert_command(
                0,
                label="No suggestions available",
                state="disabled"
            )
            self.suggestion_items.append("No suggestions")

        # Show the menu
        self.spelling_menu.tk_popup(x, y)

        # Set up menu close handler to remove temporary highlighting
        def on_menu_close():
            if hasattr(self, 'current_spellcheck_word'):
                widget = self.current_spellcheck_word['widget']
                widget.tag_remove(self.spell_tags['misspelled_highlight'], "1.0", "end")

        # Bind to menu close
        self.spelling_menu.bind("<Unmap>", lambda e: on_menu_close())

    def replace_spelling_suggestion(self, suggestion):
        """Replace misspelled word with suggestion"""
        if not hasattr(self, 'current_spellcheck_word'):
            return

        word_info = self.current_spellcheck_word
        widget = word_info['widget']

        # Replace the word
        widget.delete(word_info['start'], word_info['end'])
        widget.insert(word_info['start'], suggestion)

        # Remove highlighting
        widget.tag_remove(self.spell_tags['misspelled_highlight'], "1.0", "end")

        # Re-check spelling
        self.check_spelling()

    def ignore_all_occurrences(self):
        """Ignore all occurrences of the current word"""
        if not hasattr(self, 'current_spellcheck_word'):
            return

        word = self.current_spellcheck_word['word']
        widget = self.current_spellcheck_word['widget']

        # Add to spell checker's known words
        self.spell_checker.word_frequency.add(word.lower())

        # Remove all instances of this word from misspelled tags
        content = widget.get("1.0", "end")
        start_index = "1.0"

        while True:
            start_index = widget.search(rf"\y{word}\y", start_index,
                                      stopindex="end", regexp=True)
            if not start_index:
                break
            end_index = f"{start_index}+{len(word)}c"

            # Remove misspelled tag from this occurrence
            widget.tag_remove(self.spell_tags['misspelled'], start_index, end_index)
            widget.tag_remove(self.spell_tags['misspelled_highlight'], start_index, end_index)

            start_index = end_index

        self.write(f"✓ Added '{word}' to dictionary (ignored everywhere)\n", "green")

    def add_to_dictionary(self):
        """Add current word to custom dictionary"""
        if not hasattr(self, 'current_spellcheck_word'):
            return

        word = self.current_spellcheck_word['word']
        widget = self.current_spellcheck_word['widget']

        # Add to spell checker
        self.spell_checker.word_frequency.add(word.lower())

        # Remove highlighting for this word
        widget.tag_remove(self.spell_tags['misspelled'],
                         self.current_spellcheck_word['start'],
                         self.current_spellcheck_word['end'])
        widget.tag_remove(self.spell_tags['misspelled_highlight'],
                         self.current_spellcheck_word['start'],
                         self.current_spellcheck_word['end'])

        self.write(f"✓ Added '{word}' to dictionary\n", "green")

    def show_spellcheck_settings(self):
        """Show spell check settings dialog"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Spell Check Settings")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 400) // 2
        y = (dialog.winfo_screenheight() - 300) // 2
        dialog.geometry(f"+{x}+{y}")

        # Language selection
        lang_frame = ctk.CTkFrame(dialog)
        lang_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(lang_frame, text="Language:", font=("Arial", 12, "bold")).pack(anchor="w")

        lang_var = ctk.StringVar(value=self.current_language_name)
        lang_menu = ctk.CTkOptionMenu(
            lang_frame,
            values=list(self.available_languages.keys()),
            variable=lang_var,
            width=200
        )
        lang_menu.pack(fill="x", pady=5)

        # Options frame
        options_frame = ctk.CTkFrame(dialog)
        options_frame.pack(fill="x", padx=20, pady=10)

        # Case sensitivity option
        case_sensitive_var = ctk.BooleanVar(value=self.case_sensitive)
        case_check = ctk.CTkCheckBox(
            options_frame,
            text="Case sensitive spell checking",
            variable=case_sensitive_var
        )
        case_check.pack(anchor="w", pady=5)

        # Ignore words with numbers
        ignore_numbers_var = ctk.BooleanVar(value=self.ignore_numbers)
        numbers_check = ctk.CTkCheckBox(
            options_frame,
            text="Ignore words containing numbers",
            variable=ignore_numbers_var
        )
        numbers_check.pack(anchor="w", pady=5)

        # Buttons frame
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(fill="x", padx=20, pady=20)

        def apply_settings():
            new_language = lang_var.get()
            if new_language != self.current_language_name:
                self.change_spellcheck_language(new_language)

            # Update settings
            self.case_sensitive = case_sensitive_var.get()
            self.ignore_numbers = ignore_numbers_var.get()

            dialog.destroy()
            # Re-check with new settings
            self.check_spelling()

        ctk.CTkButton(
            button_frame,
            text="Apply",
            command=apply_settings,
            width=100
        ).pack(side="right", padx=5)

        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            width=100
        ).pack(side="right", padx=5)

    # RESTORE THE ORIGINAL REAL-TIME SPELL CHECKING
    def delayed_spell_check(self, event=None):
        """Delayed spell check to avoid conflict with syntax highlighting"""
        if not self.spell_checking_enabled:
            return

        # Cancel any pending spell check
        if hasattr(self, '_spell_check_timer'):
            self.after_cancel(self._spell_check_timer)

        # Schedule spell check to run after syntax highlighting
        self._spell_check_timer = self.after(100, self.check_spelling)

    def perform_initial_spell_check(self):
        """Perform initial spell check after UI is fully loaded"""
        if self.spell_checking_enabled:
            self.check_spelling()
            print("✓ Initial spell check completed")

     #----------------------------------------------------------------


    def on_text_click(self, event):
        """Handle text click to show anchored spelling suggestions"""
        if not self.spell_checking_enabled:
            return

        widget = event.widget
        index = widget.index(f"@{event.x},{event.y}")

        # Check if we're clicking on a misspelled word
        if "misspelled" in widget.tag_names(index):
            # Get the word under cursor
            word_start = widget.index(f"{index} wordstart")
            word_end = widget.index(f"{index} wordend")
            word = widget.get(word_start, word_end)

            # Remove any previous highlighting
            widget.tag_remove("misspelled_anchor", "1.0", "end")

            # Highlight the clicked word
            widget.tag_add("misspelled_anchor", word_start, word_end)

            # Show anchored tooltip
            self.show_anchored_tooltip(widget, word, word_start, word_end, event.x_root, event.y_root)
        else:
            # Clicked on normal text, hide tooltip
            self.hide_anchored_tooltip()

    def on_text_hover(self, event):
        """Show spelling suggestions when hovering over misspelled words"""
        if not self.spell_checking_enabled:
            return

        widget = event.widget
        index = widget.index(f"@{event.x},{event.y}")

        # Check if we're hovering over a misspelled word
        if "misspelled" in widget.tag_names(index):
            # Get the word under cursor
            word_start = widget.index(f"{index} wordstart")
            word_end = widget.index(f"{index} wordend")
            word = widget.get(word_start, word_end)

            # Only show tooltip if it's a different word than current
            if word != self.current_hover_word:
                self.current_hover_word = word
                self.show_spelling_tooltip(widget, word, event.x_root, event.y_root)
        else:
            self.hide_spelling_tooltip()

    def show_spelling_tooltip(self, widget, word, x, y):
        """Show spelling suggestions in a tooltip bubble"""
        # Hide any existing tooltip
        self.hide_spelling_tooltip()

        # Get spelling suggestions
        try:
            suggestions = list(self.spell_checker.candidates(word))[:5]
            if not suggestions:
                # Fall back to correction method
                correction = self.spell_checker.correction(word)
                if correction and correction != word:
                    suggestions = [correction]
        except Exception:
            suggestions = []

        if not suggestions:
            return

        # Create tooltip window
        self.spelling_tooltip = tk.Toplevel(self)
        self.spelling_tooltip.wm_overrideredirect(True)
        self.spelling_tooltip.wm_geometry(f"+{x+15}+{y+15}")
        self.spelling_tooltip.configure(bg='#ffffe0', relief='solid', borderwidth=1)

        # Make tooltip semi-transparent (if supported)
        try:
            self.spelling_tooltip.attributes('-alpha', 0.95)
        except:
            pass

        # Add title
        title_label = tk.Label(
            self.spelling_tooltip,
            text=f"Suggestions for '{word}':",
            bg='#ffffe0',
            fg='#333333',
            font=("Arial", 9, "bold"),
            justify='left'
        )
        title_label.pack(padx=8, pady=(8, 4), anchor='w')

        # Add suggestions
        for i, suggestion in enumerate(suggestions):
            suggestion_frame = tk.Frame(self.spelling_tooltip, bg='#ffffe0')
            suggestion_frame.pack(fill='x', padx=8, pady=2)

            suggestion_label = tk.Label(
                suggestion_frame,
                text=suggestion,
                bg='#ffffe0',
                fg='#0066cc',
                font=("Arial", 9),
                cursor="hand2",
                justify='left'
            )
            suggestion_label.pack(side='left', anchor='w')

            # Bind click event to replace word
            suggestion_label.bind("<Button-1>",
                lambda e, s=suggestion, ws=word_start, we=word_end:
                self.replace_word_from_tooltip(s, ws, we))

            # Add hover effect
            suggestion_label.bind("<Enter>",
                lambda e, lbl=suggestion_label: lbl.configure(bg='#e6f3ff'))
            suggestion_label.bind("<Leave>",
                lambda e, lbl=suggestion_label: lbl.configure(bg='#ffffe0'))

        # Add ignore option
        ignore_frame = tk.Frame(self.spelling_tooltip, bg='#ffffe0')
        ignore_frame.pack(fill='x', padx=8, pady=(4, 8))

        ignore_label = tk.Label(
            ignore_frame,
            text="Ignore",
            bg='#ffffe0',
            fg='#666666',
            font=("Arial", 8, "italic"),
            cursor="hand2"
        )
        ignore_label.pack(side='left', anchor='w')
        ignore_label.bind("<Button-1>", lambda e: self.hide_spelling_tooltip())

    def hide_spelling_tooltip(self, event=None):
        """Hide the spelling suggestion tooltip"""
        if self.spelling_tooltip:
            try:
                self.spelling_tooltip.destroy()
            except:
                pass
            self.spelling_tooltip = None
        self.current_hover_word = None

    def replace_word_from_tooltip(self, suggestion, word_start, word_end):
        """Replace word from tooltip selection"""
        focused_widget = self.focus_get()
        if focused_widget in [self.content_editor._textbox, self.notes_editor._textbox]:
            focused_widget.delete(word_start, word_end)
            focused_widget.insert(word_start, suggestion)
            self.hide_spelling_tooltip()
            # Re-check spelling after replacement
            self.check_spelling()

    def show_spelling_suggestions(self, event):
        """Show spelling suggestions on right-click"""
        if not self.spell_checking_enabled:
            return

        widget = event.widget
        # Get the word under the cursor
        index = widget.index(f"@{event.x},{event.y}")
        word_start = widget.index(f"{index} wordstart")
        word_end = widget.index(f"{index} wordend")
        word = widget.get(word_start, word_end)

        # Skip if word is too short or not misspelled
        if len(word) < 4 or word.lower() in self.spell_checker:
            return

        # Highlight the word and show anchored tooltip
        widget.tag_remove("misspelled_anchor", "1.0", "end")
        widget.tag_add("misspelled_anchor", word_start, word_end)

        self.show_anchored_tooltip(widget, word, word_start, word_end, event.x_root, event.y_root)

    def replace_word(self, widget, start, end, replacement):
        """Replace a word with the selected suggestion"""
        widget.delete(start, end)
        widget.insert(start, replacement)
        # Remove highlight
        widget.tag_remove("misspelled_highlight", start, end)
        # Re-check spelling after replacement
        self.check_spelling()


    def hide_spelling_menu(self, event):
        """Hide spelling menu when clicking elsewhere"""
        try:
            self.spelling_menu.unpost()
        except:
            pass
        # Remove highlight
        for widget in [self.content_editor._textbox, self.notes_editor._textbox]:
            widget.tag_remove("misspelled_highlight", "1.0", "end")


#-------------------------------------------------------------------------------------
    def setup_logo(self):
        """Initialize logo with correct path handling"""
        try:
            # Get package paths
            package_root, resources_dir = setup_paths()

            # Possible logo locations in priority order
            possible_paths = [
                resources_dir / 'airis4d_logo.png',  # In resources directory
                package_root / 'icons' / 'airis4d_logo.png',  # In icons subdirectory
                package_root / 'airis4d_logo.png',  # In package root
                Path(__file__).parent / 'icons' / 'airis4d_logo.png',  # Relative to script
                Path(os.path.abspath(sys.prefix)) / 'share' / 'bsg-ide' / 'resources' / 'airis4d_logo.png',  # System share
                Path.home() / '.local' / 'share' / 'bsg-ide' / 'resources' / 'airis4d_logo.png'  # User data
            ]

            # Try each location
            for path in possible_paths:
                if path.exists():
                    try:
                        self.logo_image = ctk.CTkImage(
                            light_image=Image.open(path),
                            dark_image=Image.open(path),
                            size=(50, 50)
                        )
                        self.has_logo = True
                        self.logo_dir = path.parent
                        print(f"✓ Found logo at: {path}")
                        break
                    except Exception as e:
                        print(f"Warning: Could not load logo from {path}: {e}")
                        continue

            if not self.has_logo:
                print("Warning: Using ASCII logo - no image found in:")
                for path in possible_paths:
                    print(f"  - {path}")
                self.has_logo = False

        except Exception as e:
            print(f"Warning: Could not setup logo: {str(e)}")
            self.has_logo = False
#---------------------------------------------------------------------------------------

    def create_widgets(self):
        """Create and initialize all UI widgets"""
        try:
            # Create menu frame first
            #self.setup_top_menu()

            # Configure grid
            self.grid_columnconfigure(1, weight=1)
            self.grid_rowconfigure(0, weight=1)

            # Create main components
            self.create_sidebar()
            self.create_main_editor()
            self.create_toolbar()
            self.create_context_menu()
            self.create_footer()

            # Create terminal after other UI elements
            self.create_terminal()

            # Setup output redirection after terminal creation
            self.setup_output_redirection()

            # Initialize variables
            self.current_file = None
            self.slides = []
            self.current_slide_index = -1

            # Setup keyboard shortcuts
            self.setup_keyboard_shortcuts()

            # Setup Python paths
            setup_python_paths()

            # Adjust grid weights to accommodate terminal
            self.grid_rowconfigure(1, weight=3)  # Main editor
            self.grid_rowconfigure(4, weight=1)  # Terminal

            # Initialize syntax highlighter
            self.syntax_highlighter = None
            if hasattr(self, 'content_editor'):
                self.syntax_highlighter = BeamerSyntaxHighlighter(self.content_editor)

            # Create notes highlighter if notes editor exists
            if hasattr(self, 'notes_editor'):
                self.notes_highlighter = BeamerSyntaxHighlighter(self.notes_editor)

        except Exception as e:
            print(f"Error creating widgets: {str(e)}")
            raise

    def create_recent_files_menu(self):
        """Create recent files menu with error handling"""
        try:
            recent_menu = tk.Menu(self.menu_frame, tearoff=0)
            for filepath in self.session_data['recent_files']:
                if os.path.exists(filepath):
                    recent_menu.add_command(
                        label=os.path.basename(filepath),
                        command=lambda f=filepath: self.load_file(f)
                    )

            if recent_menu.index('end') is not None:  # Only add if menu has items
                self.menu_frame.add_cascade(label="Recent Files", menu=recent_menu)
        except Exception as e:
            print(f"Warning: Could not create recent files menu: {str(e)}")

    def print_loading_summary(self):
        """Print a summary of loaded slides for debugging"""
        logger.info("\n" + "="*60)
        logger.info("LOADING SUMMARY")
        logger.info("="*60)

        for idx, slide in enumerate(self.slides):
            logger.info(f"\nSlide {idx + 1}:")
            logger.info(f"  Title: {slide.get('title')}")
            logger.info(f"  Media: {slide.get('media')} (masked: {slide.get('_media_masked', False)})")
            logger.info(f"  Content lines: {len(slide.get('content', []))}")
            logger.info(f"    Hidden indices: {slide.get('_hidden_content_indices', [])}")
            for i, line in enumerate(slide.get('content', [])):
                is_hidden = i in slide.get('_hidden_content_indices', [])
                logger.info(f"    [{i}] {'[HIDDEN]' if is_hidden else '[VISIBLE]'} {line[:50]}")
            logger.info(f"  Notes lines: {len(slide.get('notes', []))}")
            logger.info(f"    Hidden note indices: {slide.get('_hidden_note_indices', [])}")
            logger.info(f"  Fully masked: {slide.get('_fully_masked', False)}")

        logger.info("="*60)

    def update_recent_files(self, filepath):
        """Update recent files list"""
        if filepath not in self.session_data['recent_files']:
            self.session_data['recent_files'].append(filepath)
        if len(self.session_data['recent_files']) > 10:
            self.session_data['recent_files'].pop(0)
        self.create_recent_files_menu()

    def on_window_configure(self, event=None):
        """Track window size and position changes"""
        if event and event.widget == self:
            self.session_data['window_size'] = {
                'width': self.winfo_width(),
                'height': self.winfo_height()
            }
            self.session_data['window_position'] = {
                'x': self.winfo_x(),
                'y': self.winfo_y()
            }

    def on_closing(self):
        """Handle window closing with error handling"""
        try:
            if self.session_manager:
                # Update session data
                self.session_data.update({
                    'last_file': self.current_file,
                    'working_directory': os.getcwd()
                })

                # Save session
                self.session_manager.save_session(self.session_data)
        except Exception as e:
            print(f"Warning: Could not save session on exit: {str(e)}")
        finally:
            # Always close window
            self.destroy()

    def load_file(self, filename: str) -> None:
        """Load presentation from file with full preservation of masked content"""
        try:
            logger.info(f"Loading file: {filename}")

            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            # Reset slides
            self.slides = []
            self.current_slide_index = -1

            # Extract presentation info
            import re
            for key in self.presentation_info:
                pattern = f"\\\\{key}{{(.*?)}}"
                match = re.search(pattern, content)
                if match:
                    self.presentation_info[key] = match.group(1)

            # Parse slides by finding all \\title blocks (including commented ones)
            # Use a more direct approach: split by newlines and rebuild slides
            lines = content.split('\n')

            current_slide = None
            in_content = False
            in_notes = False
            content_lines = []
            notes_lines = []
            title = ""
            media = ""
            media_masked = False

            masked_slides_count = 0
            total_masked_lines = 0

            for line_num, line in enumerate(lines):
                # Check for title line (may have % before it)
                title_match = re.match(r'^(%?)\\title\s+(.+)$', line)
                if title_match:
                    # Save previous slide if exists
                    if current_slide is not None or title:
                        # Create slide from accumulated data
                        slide_data = self._create_slide_from_parsed_data(
                            title, media, media_masked,
                            content_lines, notes_lines,
                            is_fully_masked, masked_slides_count
                        )
                        if slide_data:
                            self.slides.append(slide_data)
                            if is_fully_masked:
                                masked_slides_count += 1

                    # Start new slide
                    title_masked = title_match.group(1) == '%'
                    title = title_match.group(2).strip()
                    is_fully_masked = title_masked
                    content_lines = []
                    notes_lines = []
                    media = ""
                    media_masked = False
                    in_content = False
                    in_notes = False

                    if title_masked:
                        logger.info(f"Found masked title: {title}")
                    continue

                # Check for Content begin
                content_begin_match = re.match(r'^(%?)\\begin{Content}$', line)
                if content_begin_match:
                    in_content = True
                    content_masked = content_begin_match.group(1) == '%'
                    if content_masked:
                        is_fully_masked = True
                    continue

                # Check for Content end
                if re.match(r'^%?\\end{Content}$', line):
                    in_content = False
                    continue

                # Check for Notes begin
                notes_begin_match = re.match(r'^(%?)\\begin{Notes}$', line)
                if notes_begin_match:
                    in_notes = True
                    notes_masked = notes_begin_match.group(1) == '%'
                    if notes_masked:
                        is_fully_masked = True
                    continue

                # Check for Notes end
                if re.match(r'^%?\\end{Notes}$', line):
                    in_notes = False
                    continue

                # Process content lines
                if in_content and line.strip():
                    # Check if line is masked
                    is_line_masked = line.lstrip().startswith('%')
                    clean_line = re.sub(r'^\s*%\s*', '', line) if is_line_masked else line

                    # Check for media directive (first line that starts with \)
                    if not media and clean_line.strip().startswith('\\'):
                        media = clean_line.strip()
                        media_masked = is_line_masked
                        if media_masked:
                            total_masked_lines += 1
                        logger.info(f"Found media in slide: {media} (masked: {media_masked})")
                    elif clean_line.strip():
                        content_lines.append(clean_line.rstrip())
                        if is_line_masked:
                            total_masked_lines += 1
                            logger.info(f"  Masked content line: {clean_line[:50]}")

                # Process notes lines
                if in_notes and line.strip():
                    is_note_masked = line.lstrip().startswith('%')
                    clean_note = re.sub(r'^\s*%\s*', '', line) if is_note_masked else line
                    if clean_note.strip():
                        notes_lines.append(clean_note.rstrip())
                        if is_note_masked:
                            total_masked_lines += 1
                            logger.info(f"  Masked note line: {clean_note[:50]}")

            # Don't forget the last slide
            if title or content_lines or notes_lines:
                slide_data = self._create_slide_from_parsed_data(
                    title, media, media_masked,
                    content_lines, notes_lines,
                    is_fully_masked, masked_slides_count
                )
                if slide_data:
                    self.slides.append(slide_data)
                    if is_fully_masked:
                        masked_slides_count += 1

            if self.slides:
                self.current_slide_index = 0
                self.load_slide(0)
                logger.info(f"Total slides loaded: {len(self.slides)}")
            else:
                logger.error("No slides were loaded!")
                messagebox.showerror("Error", "No slides could be loaded from the file!")
                return

            self.update_slide_list()

            # Show loading summary
            summary_parts = [f"✓ Loaded {len(self.slides)} slides"]
            if masked_slides_count > 0:
                summary_parts.append(f"{masked_slides_count} fully masked")
            if total_masked_lines > 0:
                summary_parts.append(f"{total_masked_lines} masked lines")

            self.write(" | ".join(summary_parts) + "\n", "green")

            if masked_slides_count > 0 or total_masked_lines > 0:
                self.write("ℹ Masked content is shown with strikethrough\n", "yellow")
                self.write("  • Click on masked line + Ctrl+Delete to unmask\n", "cyan")

        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.write(f"✗ {error_msg}\n", "red")
            messagebox.showerror("Error", error_msg)

#--------------------------------------------------------------------------------
    def ide_callback(self, action, data):
        """Enhanced IDE callback handler with proper \\None handling"""
        if action == "update_current_slide":
            # Update slide title
            self.title_entry.delete(0, 'end')
            self.title_entry.insert(0, data.get('title', ''))

            # Ensure proper highlight in slide list
            self.current_slide_index = data.get('index', 0)
            self.update_slide_list()
            self.highlight_current_slide()

            # Important: Reset media when changing slides
            self.media_entry.delete(0, 'end')

        elif action == "update_content":
            # Update content editor, ensuring clean state
            self.content_editor.delete('1.0', 'end')
            for line in data.get('content', []):
                if line and line.strip():  # Only add non-empty lines
                    self.content_editor.insert('end', f"{line}\n")

        elif action == "update_media":
            # Update media entry with proper \None handling
            self.media_entry.delete(0, 'end')
            media = data.get('media')

            # Handle different media cases
            if media is None or media == "\\None":
                self.media_entry.insert(0, "\\None")
            elif media:  # Only insert if there's actual media content
                self.media_entry.insert(0, media)

        elif action == "show_current_slide":
            # Show complete slide with proper state reset
            self.current_slide_index = data.get('index', 0)

            # Clear everything first
            self.title_entry.delete(0, 'end')
            self.media_entry.delete(0, 'end')
            self.content_editor.delete('1.0', 'end')

            # Update title
            if title := data.get('title'):
                self.title_entry.insert(0, title)

            # Update media with explicit \None handling
            media = data.get('media')
            if media is None or media == "\\None":
                self.media_entry.insert(0, "\\None")
            elif media:
                self.media_entry.insert(0, media)

            # Update content
            content = data.get('content', [])
            if content:
                for line in content:
                    if line and line.strip():
                        self.content_editor.insert('end', f"{line}\n")

            # Update slide list display
            self.update_slide_list()
            self.slide_list.see(f"{self.current_slide_index + 1}.0")
            self.highlight_current_slide()

        elif action == "navigate_to_slide":
            # Navigate to specific slide with complete state reset
            index = data.get('index', 0)
            if 0 <= index < len(self.slides):
                # Clear current state
                self.media_entry.delete(0, 'end')
                self.content_editor.delete('1.0', 'end')

                self.current_slide_index = index
                self.load_slide(index)
                self.update_slide_list()

                if data.get('focus'):
                    self.slide_list.see(f"{index + 1}.0")
                    self.highlight_current_slide()

        elif action == "error":
            # Show error in terminal
            if hasattr(self, 'terminal'):
                self.terminal.write(f"Error: {data.get('message', 'Unknown error')}\n", "red")

    def load_slide(self, index):
        """Enhanced load_slide with proper hidden content display"""
        if 0 <= index < len(self.slides):
            slide = self.slides[index]

            # Clear all fields first
            self.title_entry.delete(0, 'end')
            self.media_entry.delete(0, 'end')
            self.content_editor.delete('1.0', 'end')
            self.notes_editor.delete('1.0', 'end')

            # Update title
            title_text = slide.get('title', '')
            self.title_entry.insert(0, title_text)

            # Update media with masking if needed
            media = slide.get('media', '')
            media_masked = slide.get('_media_masked', False)

            if not media or media == "\\None":
                self.media_entry.insert(0, "\\None")
            elif media_masked:
                self.media_entry.insert(0, f"% {media}")
            else:
                self.media_entry.insert(0, media)

            # Configure tags for hidden content (dimmed/strikethrough)
            # Remove existing tag configurations first to avoid conflicts
            try:
                self.content_editor._textbox.tag_delete('hidden_content')
                self.notes_editor._textbox.tag_delete('hidden_note')
            except:
                pass

            self.content_editor._textbox.tag_config('hidden_content',
                                                      foreground='#888888',
                                                      font=('TkFixedFont', 10, 'overstrike'))
            self.notes_editor._textbox.tag_config('hidden_note',
                                                   foreground='#888888',
                                                   font=('TkFixedFont', 10, 'overstrike'))

            # Get hidden indices
            hidden_content_indices = set(slide.get('_hidden_content_indices', []))
            hidden_note_indices = set(slide.get('_hidden_note_indices', []))

            # DEBUG: Print hidden indices to verify
            print(f"Loading slide {index}: Hidden content indices: {hidden_content_indices}")
            print(f"Loading slide {index}: Content lines: {slide.get('content', [])}")

            # Update content with proper masking display
            for i, item in enumerate(slide.get('content', [])):
                if item.strip():
                    # Check if this line should be hidden
                    is_hidden = i in hidden_content_indices

                    if is_hidden:
                        # Insert with % marker and apply hidden tag
                        self.content_editor.insert('end', f"% {item}\n")
                        # Apply tag to the line we just inserted
                        line_start = f"{self.content_editor.index('end-2l')}"
                        line_end = f"{self.content_editor.index('end-1c')}"
                        self.content_editor._textbox.tag_add('hidden_content', line_start, line_end)
                    else:
                        self.content_editor.insert('end', f"{item}\n")

            # Update notes with proper masking display
            for i, note in enumerate(slide.get('notes', [])):
                if note.strip():
                    is_hidden = i in hidden_note_indices

                    if is_hidden:
                        self.notes_editor.insert('end', f"% {note}\n")
                        line_start = f"{self.notes_editor.index('end-2l')}"
                        line_end = f"{self.notes_editor.index('end-1c')}"
                        self.notes_editor._textbox.tag_add('hidden_note', line_start, line_end)
                    else:
                        self.notes_editor.insert('end', f"{note}\n")

            # Add placeholder text if empty
            if not slide.get('content'):
                self.content_editor.insert('end', "% No content for this slide\n")

            if not slide.get('notes'):
                self.notes_editor.insert('end', "% No notes for this slide\n")

            # Refresh syntax highlighting
            if hasattr(self, 'syntax_highlighter') and self.syntax_highlighter.active:
                self.syntax_highlighter.highlight()

    def format_tikz_for_output(self, media_content: str) -> str:
        """Format TikZ code for proper output in the text file"""
        if "% TikZ Diagram:" in media_content:
            # Extract just the TikZ code part
            lines = media_content.split('\n')
            tikz_code = []
            in_tikz = False

            for line in lines:
                if '\\begin{tikzpicture}' in line:
                    in_tikz = True
                if in_tikz:
                    tikz_code.append(line)
                if '\\end{tikzpicture}' in line:
                    in_tikz = False

            return '\n'.join(tikz_code)
        return media_content

    def on_line_click_to_unmask(self, event):
        """Handle clicking on a masked line to show unmask option"""
        if not hasattr(self, 'spell_checking_enabled'):
            return

        focused_widget = event.widget

        # Check if this is one of our editors
        if focused_widget not in [self.content_editor._textbox, self.notes_editor._textbox]:
            return

        index = focused_widget.index(f"@{event.x},{event.y}")

        # Check if clicking on a hidden line
        if 'hidden_content' in focused_widget.tag_names(index) or 'hidden_note' in focused_widget.tag_names(index):
            # Show a tooltip or context menu with unmask option
            self.show_unmask_prompt(focused_widget, index)

    def show_unmask_prompt(self, widget, index):
        """Show prompt to unmask a line"""
        # Get the line content
        line_start = widget.index(f"{index} linestart")
        line_end = widget.index(f"{index} lineend")
        line_content = widget.get(line_start, line_end)

        # Create a popup menu
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Unmask this line",
                         command=lambda: self.unmask_line_at_index(widget, line_start, line_end))
        menu.add_command(label="Unmask all lines in this slide",
                         command=lambda: self.unmask_all_lines_in_current_slide())
        menu.add_separator()
        menu.add_command(label="Cancel")

        # Show the menu at mouse position
        x = widget.winfo_pointerx()
        y = widget.winfo_pointery()
        menu.tk_popup(x, y)

    def unmask_line_at_index(self, widget, line_start, line_end):
        """Unmask a specific line"""
        line_content = widget.get(line_start, line_end)
        # Remove the % prefix
        unmasked = re.sub(r'^\s*%\s*', '', line_content)
        widget.delete(line_start, line_end)
        widget.insert(line_start, unmasked)

        # Update the slide data
        self.save_current_slide()
        self.write("✓ Line unmasked\n", "green")

    def unmask_all_lines_in_current_slide(self):
        """Unmask all hidden lines in the current slide"""
        if self.current_slide_index < 0:
            return

        # Process content editor
        content_text = self.content_editor.get('1.0', 'end-1c')
        content_lines = content_text.split('\n')
        unmasked_content = []

        for line in content_lines:
            if line.lstrip().startswith('%'):
                unmasked = re.sub(r'^\s*%\s*', '', line)
                unmasked_content.append(unmasked)
            else:
                unmasked_content.append(line)

        # Update content editor
        self.content_editor.delete('1.0', 'end')
        self.content_editor.insert('1.0', '\n'.join(unmasked_content))

        # Process notes editor similarly
        notes_text = self.notes_editor.get('1.0', 'end-1c')
        notes_lines = notes_text.split('\n')
        unmasked_notes = []

        for line in notes_lines:
            if line.lstrip().startswith('%'):
                unmasked = re.sub(r'^\s*%\s*', '', line)
                unmasked_notes.append(unmasked)
            else:
                unmasked_notes.append(line)

        self.notes_editor.delete('1.0', 'end')
        self.notes_editor.insert('1.0', '\n'.join(unmasked_notes))

        # Clear hidden indices in slide data
        if 0 <= self.current_slide_index < len(self.slides):
            self.slides[self.current_slide_index]['_hidden_content_indices'] = []
            self.slides[self.current_slide_index]['_hidden_note_indices'] = []
            self.slides[self.current_slide_index]['_media_masked'] = False

        self.save_current_slide()
        self.write(f"✓ All lines unmasked in slide {self.current_slide_index + 1}\n", "green")

#--------------------------------------------------------------------------------------------------------------------
    def setup_output_redirection(self):
        """Set up output redirection to terminal"""
        if hasattr(self, 'terminal'):
            sys.stdout = SimpleRedirector(self.terminal, "white")
            sys.stderr = SimpleRedirector(self.terminal, "red")

    # Method to update working directory
    def update_terminal_directory(self, directory: str) -> None:
        """Update terminal working directory"""
        if hasattr(self, 'terminal'):
            self.terminal.set_working_directory(directory)
#---------------------------------------------------------------------------New run pdflatex ----------------
    # First, add an input method to handle terminal input
    def terminal_input(self, prompt: str) -> str:
        """Get input from terminal with prompt"""
        try:
            self.write(prompt, "yellow")
            # Use terminal's built-in input handling
            future_input = []
            input_ready = threading.Event()

            def on_input(text):
                future_input.append(text)
                input_ready.set()

            # Store current input handler
            old_handler = self.terminal._handle_input

            # Set temporary input handler
            def temp_handler(event):
                text = self.terminal.display._textbox.get("insert linestart", "insert lineend")
                if text.startswith("$ "):
                    text = text[2:].strip()
                    self.terminal.write("\n")
                    on_input(text)
                    return "break"
                return "break"

            self.terminal.display.bind("<Return>", temp_handler)

            # Wait for input
            input_ready.wait()

            # Restore original handler
            self.terminal.display.bind("<Return>", old_handler)

            return future_input[0] if future_input else ""

        except Exception as e:
            self.write(f"Error getting input: {str(e)}\n", "red")
            return ""

    def navigate_slides(self, event):
        """Handle keyboard navigation between slides"""
        if not self.slides:
            return

        # Save current slide before navigation
        self.save_current_slide()

        if event.keysym == 'Up':
            if self.current_slide_index > 0:
                self.current_slide_index -= 1
        elif event.keysym == 'Down':
            if self.current_slide_index < len(self.slides) - 1:
                self.current_slide_index += 1
        elif event.keysym == 'Left':
            if self.current_slide_index > 0:
                self.current_slide_index -= 1
        elif event.keysym == 'Right':
            if self.current_slide_index < len(self.slides) - 1:
                self.current_slide_index += 1

        # Load the new slide
        self.load_slide(self.current_slide_index)
        self.update_slide_list()

        # Ensure the current slide is visible in the list
        line_number = self.current_slide_index + 1
        self.slide_list.see(f"{line_number}.0")

        # Update visual highlighting
        self.highlight_current_slide()

        return "break"  # Prevent default handling
    #----------------------------------------------------------------
    def write_to_terminal(self, text: str, color: str = "white") -> None:
        """Alias for write method to maintain compatibility"""
        self.write(text, color)


    def create_terminal(self) -> None:
        """Create terminal initially hidden"""
        # Create terminal instance
        self.terminal = InteractiveTerminal(self, initial_directory=os.getcwd())
        self.terminal.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.grid_rowconfigure(4, weight=0)  # Initially no weight

        # Hide terminal initially
        self.terminal.grid_remove()

        # Set up redirections
        sys.stdout = SimpleRedirector(self.terminal, "white")
        sys.stderr = SimpleRedirector(self.terminal, "red")

        # Store process reference
        self.current_process = None

    def toggle_terminal(self) -> None:
        """Toggle terminal visibility"""
        self.terminal_visible = not self.terminal_visible

        if self.terminal_visible:
            # Show terminal
            self.terminal.grid()
            self.grid_rowconfigure(4, weight=1)  # Give weight when visible
            self.terminal_button.configure(text="▼ Hide Terminal")

            # Optionally resize window to accommodate terminal
            current_height = self.winfo_height()
            if current_height < 800:  # Minimum height with terminal
                new_height = min(800, self.winfo_screenheight() - 100)
                self.geometry(f"{self.winfo_width()}x{new_height}")
        else:
            # Hide terminal
            self.terminal.grid_remove()
            self.grid_rowconfigure(4, weight=0)  # Remove weight when hidden
            self.terminal_button.configure(text="▲ Show Terminal")

            # Optionally resize window back
            if self.winfo_height() > 600:  # Minimum height without terminal
                self.geometry(f"{self.winfo_width()}x600")


    def flush(self):
        """Required for stdout/stderr redirection"""
        pass

    # Replace the clear_terminal method
    def clear_terminal(self) -> None:
        """Clear terminal content"""
        if hasattr(self, 'terminal'):
            self.terminal.clear()

    def stop_compilation(self) -> None:
        """Stop current compilation process"""
        if self.current_process:
            try:
                self.current_process.terminate()
                self.write("\n[Compilation process terminated by user]\n")
            except:
                pass
            finally:
                self.current_process = None

#--------------------------------------------------------------------------------------------------------------------
    def create_footer(self):
        """Create footer with institution info and properly tracked logo"""
        # Footer frame with dark theme
        self.footer = ctk.CTkFrame(self)
        self.footer.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Left side - Logo and Institution name
        left_frame = ctk.CTkFrame(self.footer, fg_color="transparent")
        left_frame.pack(side="left", padx=10)

        # Add terminal toggle button
        self.terminal_visible = False
        self.terminal_button = ctk.CTkButton(
            left_frame,
            text="▲ Show Terminal",
            command=self.toggle_terminal,
            width=120,
            height=24,
            fg_color="#2F3542",
            hover_color="#404859"
        )
        self.terminal_button.pack(side="left", padx=10)

        # Logo (image or ASCII)
        if self.has_logo and hasattr(self, 'logo_image'):
            try:
                logo_label = ctk.CTkLabel(
                    left_frame,
                    image=self.logo_image,
                    text=""
                )
                logo_label.pack(side="left", padx=(0, 10))
            except Exception as e:
                print(f"Warning: Could not create logo label: {e}")
                self.create_ascii_logo(left_frame)
        else:
            self.create_ascii_logo(left_frame)

        # Institution name
        inst_label = ctk.CTkLabel(
            left_frame,
            text="Artificial Intelligence Research and Intelligent Systems (airis4D)",
            font=("Arial", 12, "bold"),
            text_color="#4ECDC4"
        )
        inst_label.pack(side="left", padx=10)

        # Right side - Contact and GitHub links
        links_frame = ctk.CTkFrame(self.footer, fg_color="transparent")
        links_frame.pack(side="right", padx=10)

        # Version info
        version_label = ctk.CTkLabel(
            links_frame,
            text=f"v{self.__version__}",
            font=("Arial", 10),
            text_color="#6272A4"
        )
        version_label.pack(side="left", padx=5)

        # Contact link
        contact_button = ctk.CTkButton(
            links_frame,
            text="nsp@airis4d.com",
            command=lambda: webbrowser.open("mailto:nsp@airis4d.com"),
            fg_color="transparent",
            text_color="#FFB86C",
            hover_color="#2F3542",
            height=20
        )
        contact_button.pack(side="left", padx=5)

        # Separator
        separator = ctk.CTkLabel(
            links_frame,
            text="|",
            text_color="#6272A4"
        )
        separator.pack(side="left", padx=5)

        # GitHub link
        github_button = ctk.CTkButton(
            links_frame,
            text="GitHub",
            command=lambda: webbrowser.open("https://github.com/sajeethphilip/Beamer-Slide-Generator.git"),
            fg_color="transparent",
            text_color="#FFB86C",
            hover_color="#2F3542",
            height=20
        )
        github_button.pack(side="left", padx=5)

        # License info
        license_label = ctk.CTkLabel(
            links_frame,
            text=f"({self.__license__})",
            font=("Arial", 10),
            text_color="#6272A4"
        )
        license_label.pack(side="left", padx=5)

    def create_ascii_logo(self, parent):
        """Create ASCII logo as fallback"""
        logo_label = ctk.CTkLabel(
            parent,
            text=self.logo_ascii,
            font=("Courier", 10),
            justify="left"
        )
        logo_label.pack(side="left", padx=(0, 10))

    def create_about_dialog(self) -> None:
        """Create about dialog with logo and information"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("About BeamerSlide Generator")
        dialog.geometry("400x300")

        # Center the dialog on the main window
        dialog.transient(self)
        dialog.grab_set()

        # Logo
        if self.has_logo:
            logo_label = ctk.CTkLabel(
                dialog,
                image=self.logo_image,
                text=""
            )
        else:
            logo_label = ctk.CTkLabel(
                dialog,
                text=self.logo_ascii,
                font=("Courier", 10),
                justify="left"
            )
        logo_label.pack(pady=20)

        # Information
        info_text = f"""
BeamerSlide Generator IDE
Version {self.__version__}

Created by {self.__author__}
{self.presentation_info['institution']}

{self.__license__} License
        """

        info_label = ctk.CTkLabel(
            dialog,
            text=info_text,
            font=("Arial", 12),
            justify="center"
        )
        info_label.pack(pady=20)

        # Close button
        close_button = ctk.CTkButton(
            dialog,
            text="Close",
            command=dialog.destroy
        )
        close_button.pack(pady=20)
#----------------------------------------------------------------------------------------
    def setup_top_menu(self) -> None:
        """Create top menu bar with proper spacing and visibility"""
        # Main menu container
        self.menu_frame = ctk.CTkFrame(self)
        self.menu_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Configure menu frame grid properly
        self.menu_frame.grid_columnconfigure(1, weight=1)  # Make middle section expandable

        # Left side buttons with fixed minimum widths
        left_buttons = ctk.CTkFrame(self.menu_frame, fg_color="transparent")
        left_buttons.grid(row=0, column=0, sticky="w", padx=5)

        # Add menu buttons with minimum width
        menu_buttons = [
            ("Edit Preamble", self.edit_preamble, "Edit LaTeX preamble"),
            ("Presentation Settings", self.show_settings_dialog, "Configure presentation settings"),
            ("Get Source", self.get_source_from_tex, "Extract source from TEX file"),
            ("Load TeX File", self.load_tex_file, "Load and convert Beamer TeX file"),
            ("Overwrite TeX+PDF", self.overwrite_tex_and_generate_pdf, "Convert back to TeX and generate PDF"),
        ]

        for i, (text, command, tooltip) in enumerate(menu_buttons):
            btn = ctk.CTkButton(
                left_buttons,
                text=text,
                command=command,
                width=130  # Fixed minimum width for buttons
            )
            btn.pack(side="left", padx=5)
            self.create_tooltip(btn, tooltip)

        # Right side buttons
        right_buttons = ctk.CTkFrame(self.menu_frame, fg_color="transparent")
        right_buttons.grid(row=0, column=1, sticky="e", padx=5)

        # Add syntax highlighting switch
        self.highlight_var = ctk.BooleanVar(value=True)
        self.highlight_switch = ctk.CTkSwitch(
            right_buttons,
            text="Syntax Highlighting",
            variable=self.highlight_var,
            command=self.toggle_highlighting,
            width=150  # Fixed minimum width for switch
        )
        self.highlight_switch.pack(side="right", padx=5)

    def get_source_from_tex(self) -> None:
        """Convert a tex file back to source text format using simple converter"""
        tex_file = filedialog.askopenfilename(
            filetypes=[("TeX files", "*.tex"), ("All files", "*.*")],
            title="Select TeX File to Convert"
        )

        if not tex_file:
            return

        try:
            # Use the standalone converter
            output_file = convert_beamer_tex_to_simple_text(tex_file)

            if output_file:
                messagebox.showinfo("Success", f"Source file created: {output_file}")

                # Ask if user wants to load the generated source file
                if messagebox.askyesno("Load File", "Would you like to load the generated source file?"):
                    self.load_file(output_file)
            else:
                messagebox.showerror("Error", "Failed to convert TeX file")

        except Exception as e:
            messagebox.showerror("Error", f"Error converting TeX file:\n{str(e)}")

    def extract_presentation_info(self, content: str) -> dict:
        """Extract presentation information from document body only"""
        info = {
            'title': '',
            'subtitle': '',
            'author': '',
            'institution': '',
            'short_institute': '',
            'date': '\\today'
        }

        import re

        # First isolate the document body
        doc_match = re.search(r'\\begin{document}(.*?)\\end{document}', content, re.DOTALL)
        if doc_match:
            document_content = doc_match.group(1).strip()

            # Look for title frame content
            title_frame = re.search(r'\\begin{frame}.*?\\titlepage.*?\\end{frame}\n',
                                  document_content,
                                  re.DOTALL)
            if title_frame:
                # Extract information from the title frame
                for key in info.keys():
                    pattern = f"\\\\{key}{{(.*?)}}"
                    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                    if match:
                        # Clean up LaTeX formatting
                        value = match.group(1).strip()
                        value = re.sub(r'\\textcolor{[^}]*}{([^}]*)}', r'\1', value)
                        value = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', value)
                        info[key] = value

        return info

    def extract_slides_from_tex(self, content: str) -> list:
        """Extract slides from TeX content with simplified content extraction"""
        slides = []
        import re

        # First isolate the document body
        doc_match = re.search(r'\\begin{document}(.*?)\\end{document}', content, re.DOTALL)
        if not doc_match:
            self.write("✗ Could not find document body\n", "red")
            return slides

        document_content = doc_match.group(1)

        # Find all frame blocks in the document body
        frame_blocks = re.finditer(
            r'\\begin{frame}\s*(?:\{\\Large\\textbf{([^}]*?)}\}|\{([^}]*)\})?(.*?)\\end{frame}\n',
            document_content,
            re.DOTALL
        )

        for block in frame_blocks:
            # Extract title from different possible patterns
            title = block.group(1) if block.group(1) else block.group(2) if block.group(2) else ""
            frame_content = block.group(3).strip() if block.group(3) else ""

            # If no title found in frame declaration, look for frametitle
            if not title:
                title_match = re.search(r'\\frametitle{([^}]*)}', frame_content)
                if title_match:
                    title = title_match.group(1)

            # Clean up title - remove \Large, \textbf, etc.
            if title:
                title = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', title)
            else:
                title = "Untitled Slide"

            # Skip title page frames
            if "\\titlepage" in frame_content or "\\maketitle" in frame_content:
                self.write(f"Skipping title frame\n", "yellow")
                continue

            # Extract media
            media = self.extract_media_from_frame(frame_content)

            # Extract content using simplified approach
            content_items = self.extract_content_from_frame(frame_content)

            # Extract notes
            notes = self.extract_notes_from_frame(frame_content)

            # Only add non-empty slides
            if content_items or media:
                slides.append({
                    'title': title.strip(),
                    'media': media,
                    'content': content_items,
                    'notes': notes
                })

        return slides

    def create_sidebar(self) -> None:
        """Create sidebar with slide list and controls including insert slide below and restore functions"""
        self.sidebar = ctk.CTkFrame(self)
        self.sidebar.grid(row=1, column=0, rowspan=2, sticky="nsew", padx=5, pady=5)

        # Slide list label
        ctk.CTkLabel(self.sidebar, text="Slides",
                    font=("Arial", 14, "bold")).grid(row=0, column=0, padx=5, pady=5)

        # Slide list with scroll
        self.slide_list = ctk.CTkTextbox(self.sidebar, width=180, height=300)
        self.slide_list.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Enhanced bindings for navigation
        self.slide_list.bind('<Button-1>', self.on_slide_select)
        self.slide_list.bind('<Up>', self.navigate_slides)
        self.slide_list.bind('<Down>', self.navigate_slides)
        self.slide_list.bind('<Left>', self.navigate_slides)
        self.slide_list.bind('<Right>', self.navigate_slides)
        self.bind('<Control-Up>', lambda e: self.move_slide(-1))
        self.bind('<Control-Down>', lambda e: self.move_slide(1))

        # Focus binding to enable keyboard navigation
        self.slide_list.bind('<FocusIn>', self.on_list_focus)
        self.slide_list.bind('<FocusOut>', self.on_list_unfocus)

        # Slide control buttons with enhanced tooltips
        button_data = [
            ("New Slide", self.new_slide, "Add a new slide at the end (Ctrl+N)"),
            ("Insert Below", self.insert_slide_below, "Insert a new slide below current (Ctrl+I)"),
            ("Duplicate Slide", self.duplicate_slide, "Create a copy of current slide (Ctrl+D)"),
            ("Delete Slide", self.delete_slide, "Mask/delete current slide (Del)"),
            ("Move Up", lambda: self.move_slide(-1), "Move current slide up (Ctrl+Up)"),
            ("Move Down", lambda: self.move_slide(1), "Move current slide down (Ctrl+Down)")
        ]

        for i, (text, command, tooltip) in enumerate(button_data, start=2):
            btn = ctk.CTkButton(self.sidebar, text=text, command=command)
            btn.grid(row=i, column=0, padx=5, pady=5)
            self.create_tooltip(btn, tooltip)

        # Add separator
        separator = ttk.Separator(self.sidebar, orient='horizontal')
        separator.grid(row=8, column=0, padx=5, pady=10, sticky="ew")

        # Restore and management buttons frame
        restore_frame = ctk.CTkFrame(self.sidebar)
        restore_frame.grid(row=9, column=0, padx=5, pady=5, sticky="ew")

        # Restore current slide button
        restore_btn = ctk.CTkButton(
            restore_frame,
            text="↩ Restore Slide",
            command=self.restore_deleted_slide,
            fg_color="#28a745",
            hover_color="#218838",
            height=32
        )
        restore_btn.pack(side="left", padx=2, fill="x", expand=True)
        self.create_tooltip(restore_btn, "Restore current deleted slide (Ctrl+Shift+R)")

        # Restore all deleted slides button
        restore_all_btn = ctk.CTkButton(
            restore_frame,
            text="♻ Restore All",
            command=self.restore_all_deleted_slides,
            fg_color="#17a2b8",
            hover_color="#138496",
            height=32
        )
        restore_all_btn.pack(side="left", padx=2, fill="x", expand=True)
        self.create_tooltip(restore_all_btn, "Restore all deleted slides")

        # Purge permanently button
        purge_btn = ctk.CTkButton(
            restore_frame,
            text="🗑 Purge All",
            command=self.permanently_delete_masked_slides,
            fg_color="#dc3545",
            hover_color="#c82333",
            height=32
        )
        purge_btn.pack(side="left", padx=2, fill="x", expand=True)
        self.create_tooltip(purge_btn, "Permanently delete all masked slides (cannot undo)")

        # Add undo/redo info label
        info_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        info_frame.grid(row=10, column=0, padx=5, pady=(10, 5), sticky="ew")

        info_label = ctk.CTkLabel(
            info_frame,
            text="💡 Undo: Ctrl+Z | Redo: Ctrl+Y",
            font=("Arial", 10),
            text_color="#4ECDC4"
        )
        info_label.pack()

        # Configure grid weights for proper resizing
        self.sidebar.grid_rowconfigure(1, weight=1)  # Slide list gets extra space
        self.sidebar.grid_columnconfigure(0, weight=1)

    def insert_slide_below(self) -> None:
        """Insert a new slide below the current slide"""
        # Save current slide first
        self.save_current_slide()

        new_slide = {
            'title': 'New Slide',
            'media': '',
            'content': [],
            'notes': []
        }

        # If there are no slides or current_slide_index is invalid
        if not self.slides or self.current_slide_index < 0:
            self.slides.append(new_slide)
            self.current_slide_index = 0
        else:
            # Insert after current slide
            insert_position = self.current_slide_index + 1
            self.slides.insert(insert_position, new_slide)
            self.current_slide_index = insert_position

        # Update UI
        self.update_slide_list()
        self.load_slide(self.current_slide_index)

        # Focus title entry for immediate editing
        self.title_entry.focus_set()
        self.title_entry.select_range(0, 'end')


    def on_list_focus(self, event) -> None:
        """Handle slide list focus"""
        self.highlight_current_slide()
        # Visual feedback that list is focused
        self.slide_list.configure(border_color="#4ECDC4")

    def on_list_unfocus(self, event) -> None:
        """Handle slide list losing focus"""
        # Remove focus visual feedback
        self.slide_list.configure(border_color="")

    def _create_slide_from_parsed_data(self, title, media, media_masked, content_lines, notes_lines, is_fully_masked, slide_num):
        """Create a slide dictionary from parsed data with proper hidden indices"""

        # Track which content lines are hidden
        hidden_content_indices = []
        # We need to know which lines were originally masked
        # Since we lost that info during parsing, we need to reconstruct from content_lines
        # For now, we'll assume all lines are visible unless we have a way to track

        # For fully masked slides, mark all content as hidden
        if is_fully_masked:
            hidden_content_indices = list(range(len(content_lines)))
            hidden_note_indices = list(range(len(notes_lines)))
            # Add [DELETED] prefix to title if not already there
            if not title.startswith('[DELETED]'):
                title = f"[DELETED] {title}"
        else:
            # For partially masked slides, we need to preserve which lines were masked
            # Since we lost this info, we'll need to re-parse the original file differently
            # For now, create empty lists
            hidden_content_indices = []
            hidden_note_indices = []

        slide_data = {
            'title': title,
            'media': media,
            'content': content_lines,
            'notes': notes_lines,
            '_hidden_content_indices': hidden_content_indices,
            '_hidden_note_indices': hidden_note_indices,
            '_media_masked': media_masked,
            '_fully_masked': is_fully_masked
        }

        logger.info(f"Created slide: {title[:50]} (fully_masked={is_fully_masked}, content_lines={len(content_lines)}, hidden={len(hidden_content_indices)})")
        return slide_data

    def duplicate_slide(self) -> None:
            """Duplicate the current slide"""
            if self.current_slide_index >= 0:
                # Save the current slide first to ensure we have the latest changes
                self.save_current_slide()

                # Create a deep copy of the current slide
                current_slide = self.slides[self.current_slide_index]
                new_slide = {
                    'title': f"{current_slide['title']} (Copy)",
                    'media': current_slide['media'],
                    'content': current_slide['content'].copy()  # Create a new list with the same content
                }

                # Insert the new slide after the current slide
                insert_position = self.current_slide_index + 1
                self.slides.insert(insert_position, new_slide)

                # Update the current slide index to point to the new slide
                self.current_slide_index = insert_position

                # Update the UI
                self.update_slide_list()
                self.load_slide(self.current_slide_index)

                # Show confirmation message
                messagebox.showinfo("Success", "Slide duplicated successfully!")
            else:
                messagebox.showwarning("Warning", "No slide to duplicate!")
#---------------------------------------------------------------------------------------------------
    def create_main_editor(self) -> None:
        """Create main editor area with enhanced media controls and editor options"""
        self.editor_frame = ctk.CTkFrame(self)
        self.editor_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        # Title section
        title_frame = ctk.CTkFrame(self.editor_frame)
        title_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(title_frame, text="Title:").pack(side="left", padx=5)
        self.title_entry = ctk.CTkEntry(title_frame, width=400)
        self.title_entry.pack(side="left", padx=5, fill="x", expand=True)

        # Media section
        media_frame = ctk.CTkFrame(self.editor_frame)
        media_frame.pack(fill="x", padx=5, pady=5)

        # Media label and entry
        media_label_frame = ctk.CTkFrame(media_frame)
        media_label_frame.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(media_label_frame, text="Media:").pack(side="left", padx=5)
        self.media_entry = ctk.CTkEntry(media_label_frame, width=300)
        self.media_entry.pack(side="left", padx=5, fill="x", expand=True)

        # Media buttons frame
        media_buttons = ctk.CTkFrame(media_frame)
        media_buttons.pack(side="right", padx=5)

        # Standard media buttons
        button_configs = [
            ("Local File", self.browse_media, "Browse local media files"),
            ("YouTube", self.youtube_dialog, "Add YouTube video"),
            ("Search Images", self.search_images, "Search for images online"),
            None,  # Separator
            ("📷 Camera", self.open_camera, "Capture from camera"),
            ("🖥️ Screen", self.capture_screen, "Capture screen area"),
            ("❌ No Media", lambda: self.media_entry.insert(0, "\\None"), "Create slide without media")
        ]

        for config in button_configs:
            if config is None:
                # Add separator
                ttk.Separator(media_buttons, orient="vertical").pack(side="left", padx=5, pady=5, fill="y")
                continue

            text, command, tooltip = config
            is_capture = text.startswith(('📷', '🖥️', '❌'))

            btn = ctk.CTkButton(
                media_buttons,
                text=text,
                command=command,
                width=90,
                fg_color="#4A90E2" if is_capture else None,
                hover_color="#357ABD" if is_capture else None
            )
            btn.pack(side="left", padx=2)
            self.create_tooltip(btn, tooltip)

        # Create editors container
        editors_frame = ctk.CTkFrame(self.editor_frame)
        editors_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Content editor section
        content_frame = ctk.CTkFrame(editors_frame)
        content_frame.pack(fill="both", expand=True, padx=5, pady=5)

        content_label_frame = ctk.CTkFrame(content_frame)
        content_label_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(content_label_frame, text="Content:").pack(side="left", padx=5)

        self.content_editor = ctk.CTkTextbox(content_frame, height=200)
        self.content_editor.pack(fill="both", expand=True, padx=5, pady=5)

        # Notes section
        notes_frame = ctk.CTkFrame(self.editor_frame)
        notes_frame.pack(fill="both", expand=True, padx=5, pady=5)

        notes_header = ctk.CTkFrame(notes_frame)
        notes_header.pack(fill="x", padx=5, pady=2)

        ctk.CTkLabel(notes_header, text="Presentation Notes:").pack(side="left", padx=5)

        # Notes control buttons on the right
        notes_buttons = ctk.CTkFrame(notes_header)
        notes_buttons.pack(side="right", padx=5)

        # Initialize notes mode
        self.notes_mode = tk.StringVar(value="both")

        # Notes mode buttons
        buttons_config = [
            ("slides", "Slides Only", "Generate slides without notes", "#2B87BB", "#1B5577"),
            ("notes", "Notes Only", "Generate notes only", "#27AE60", "#1A7340"),
            ("both", "Slides + Notes", "Generate slides with notes", "#8E44AD", "#5E2D73")
        ]

        for mode, text, tooltip, active_color, hover_color in buttons_config:
            btn = ctk.CTkButton(
                notes_buttons,
                text=text,
                command=lambda m=mode: self.set_notes_mode(m),
                width=100,
                fg_color=active_color if self.notes_mode.get() == mode else "gray",
                hover_color=hover_color
            )
            btn.pack(side="left", padx=2)
            self.create_tooltip(btn, tooltip)
            self.notes_buttons[mode] = {
                'button': btn,
                'active_color': active_color,
                'hover_color': hover_color
            }

        # Editor options row (new)
        editor_options = ctk.CTkFrame(notes_frame)
        editor_options.pack(fill="x", padx=5, pady=(10, 5))

        # Add syntax highlighting switch
        self.highlight_var = ctk.BooleanVar(value=True)
        self.highlight_switch = ctk.CTkSwitch(
            editor_options,
            text="Syntax Highlighting",
            variable=self.highlight_var,
            command=self.toggle_highlighting,
            width=150
        )
        self.highlight_switch.pack(side="left", padx=5)

        # Notes editor
        self.notes_editor = ctk.CTkTextbox(notes_frame, height=150)
        self.notes_editor.pack(fill="both", expand=True, padx=5, pady=5)

        # Initialize syntax highlighters
        self.syntax_highlighter = BeamerSyntaxHighlighter(self.content_editor)
        self.notes_highlighter = BeamerSyntaxHighlighter(self.notes_editor)

        # Set initial button colors
        self.update_notes_buttons(self.notes_mode.get())

        # Add help indicator
        help_frame = ctk.CTkFrame(self.editor_frame)
        help_frame.pack(fill="x", padx=5, pady=2)

        help_label = ctk.CTkLabel(
            help_frame,
            text="💡 Hover over LaTeX commands (starting with \\) for help",
            font=("Arial", 10),
            text_color="#4ECDC4"
        )
        help_label.pack(side="left")


    def check_dependencies(self) -> dict:
        """Check if required packages are installed in current environment"""
        dependencies = {
            'PIL': {'import_name': 'PIL', 'package_name': 'pillow', 'installed': False},
            'pyautogui': {'import_name': 'pyautogui', 'package_name': 'pyautogui', 'installed': False},
            'cv2': {'import_name': 'cv2', 'package_name': 'opencv-python', 'installed': False}
        }

        for dep_name, dep_info in dependencies.items():
            try:
                __import__(dep_info['import_name'])
                dep_info['installed'] = True
            except ImportError:
                dep_info['installed'] = False

        return dependencies

#-------------------------------------------------------Capture Screen ---------------------------------------
    def capture_screen(self):
        """Cross-platform screen capture with animation support"""
        try:
            import platform
            import pyautogui
            from PIL import ImageGrab
            from screeninfo import get_monitors, ScreenInfoError

            system = platform.system()

            # Get monitor info safely
            try:
                monitors = get_monitors()
                total_width = max(m.x + m.width for m in monitors)
                total_height = max(m.y + m.height for m in monitors)
                screen_left = min(m.x for m in monitors)
                screen_top = min(m.y for m in monitors)
            except (ScreenInfoError, ValueError):
                # Fallback to primary screen
                total_width = self.winfo_screenwidth()
                total_height = self.winfo_screenheight()
                screen_left = 0
                screen_top = 0

            # Create root window
            root = tk.Tk()
            root.withdraw()

            # Create overlay window
            overlay = tk.Toplevel(root)
            overlay.title("Screen Capture")

            # OS-specific setup
            if system == "Darwin":  # macOS
                overlay.attributes('-transparent', True)
                overlay.attributes('-alpha', 0.1)
            elif system == "Windows":
                overlay.attributes('-alpha', 0.01)
                overlay.attributes('-transparentcolor', 'black')
            else:  # Linux
                overlay.attributes('-type', 'splash')
                overlay.attributes('-alpha', 0.01)
                overlay.attributes('-topmost', True)

            overlay.overrideredirect(True)
            overlay.geometry(f"{total_width}x{total_height}+{screen_left}+{screen_top}")

            # Create canvas
            canvas = tk.Canvas(
                overlay,
                highlightthickness=0,
                cursor="crosshair",
                bg="black"
            )
            canvas.pack(fill='both', expand=True)

            selection = {'start_x': None, 'start_y': None, 'current_rect': None}

            def capture_area(bbox):
                """Capture screen area with OS-specific handling"""
                overlay.withdraw()
                root.update()
                time.sleep(0.2)
                try:
                    if system == "Darwin":
                        try:
                            import Quartz
                            region = Quartz.CGRectMake(bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
                            image = Quartz.CGWindowListCreateImage(
                                region,
                                Quartz.kCGWindowListOptionOnScreenOnly,
                                Quartz.kCGNullWindowID,
                                Quartz.kCGWindowImageDefault
                            )
                            width = Quartz.CGImageGetWidth(image)
                            height = Quartz.CGImageGetHeight(image)
                            data = Quartz.CGDataProviderCopyData(
                                Quartz.CGImageGetDataProvider(image))
                            return Image.frombytes("RGBA", (width, height), data)
                        except ImportError:
                            return ImageGrab.grab(bbox=bbox)
                    else:
                        return ImageGrab.grab(bbox=bbox)
                finally:
                    if not selection.get('is_cancelled'):
                        overlay.deiconify()

            def on_mouse_down(event):
                selection['start_x'] = event.x
                selection['start_y'] = event.y
                if selection.get('current_rect'):
                    canvas.delete(selection['current_rect'])

            def on_mouse_move(event):
                if selection['start_x'] is None:
                    return
                if selection.get('current_rect'):
                    canvas.delete(selection['current_rect'])
                selection['current_rect'] = canvas.create_rectangle(
                    selection['start_x'], selection['start_y'],
                    event.x, event.y,
                    outline='red', width=2
                )
                # Show dimensions
                width = abs(event.x - selection['start_x'])
                height = abs(event.y - selection['start_y'])
                canvas.delete('dimensions')
                canvas.create_text(
                    event.x + 10, event.y + 10,
                    text=f"{width}x{height}",
                    fill='red',
                    anchor='nw',
                    tags='dimensions'
                )

            def on_mouse_up(event):
                if selection['start_x'] is None:
                    return
                try:
                    # Calculate coordinates
                    x1 = min(selection['start_x'], event.x)
                    y1 = min(selection['start_y'], event.y)
                    x2 = max(selection['start_x'], event.x)
                    y2 = max(selection['start_y'], event.y)

                    # Adjust for screen offset
                    x1 += screen_left
                    y1 += screen_top
                    x2 += screen_left
                    y2 += screen_top

                    bbox = (x1, y1, x2, y2)

                    if self.capture_mode.get() == "animation":
                        frames = []
                        progress = tk.Toplevel(root)
                        progress.title("Capturing Animation")
                        progress.transient(root)
                        progress_label = tk.Label(progress, text="Capturing frames...")
                        progress_label.pack(pady=10)
                        pbar = ttk.Progressbar(progress, length=200, mode='determinate')
                        pbar.pack(pady=10)

                        try:
                            for i in range(self.frame_count.get()):
                                if selection.get('is_cancelled'):
                                    break
                                progress_label['text'] = f"Capturing frame {i+1}/{self.frame_count.get()}"
                                pbar['value'] = (i + 1) / self.frame_count.get() * 100
                                progress.update()
                                frame = capture_area(bbox)
                                if frame:
                                    frames.append(frame)
                                time.sleep(self.frame_delay.get())  # Use the specified delay
                                if i == self.frame_count.get() - 1:  # Last frame captured
                                    break

                            if frames and not selection.get('is_cancelled'):
                                # Save as GIF
                                os.makedirs('media_files', exist_ok=True)
                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                filename = f"screen_animation_{timestamp}.gif"
                                filepath = os.path.join('media_files', filename)
                                frames[0].save(
                                    filepath,
                                    save_all=True,
                                    append_images=frames[1:],
                                    duration=int(self.frame_delay.get() * 1000),  # Convert to milliseconds
                                    loop=0
                                )
                                self.media_entry.delete(0, 'end')
                                self.media_entry.insert(0, f"\\file media_files/{filename}")
                                messagebox.showinfo("Success", f"Animation saved as:\n{filename}")
                        finally:
                            progress.destroy()
                            cleanup()  # Automatically clean up after capturing frames
                    else:
                        # Single frame capture
                        screenshot = capture_area(bbox)
                        if screenshot and not selection.get('is_cancelled'):
                            os.makedirs('media_files', exist_ok=True)
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            filename = f"screen_capture_{timestamp}.png"
                            filepath = os.path.join('media_files', filename)
                            screenshot.save(filepath)
                            self.media_entry.delete(0, 'end')
                            self.media_entry.insert(0, f"\\file media_files/{filename}")
                            messagebox.showinfo("Success", f"Screenshot saved as:\n{filename}")
                except Exception as e:
                    messagebox.showerror("Error", f"Capture failed:\n{str(e)}")
                finally:
                    cleanup()

            def on_escape(event=None):
                selection['is_cancelled'] = True
                cleanup()

            def cleanup():
                try:
                    overlay.destroy()
                    root.destroy()
                    root.quit()
                except:
                    pass

            # Bind events
            canvas.bind('<Button-1>', on_mouse_down)
            canvas.bind('<B1-Motion>', on_mouse_move)
            canvas.bind('<ButtonRelease-1>', on_mouse_up)
            overlay.bind('<Escape>', on_escape)
            overlay.protocol("WM_DELETE_WINDOW", on_escape)

            root.mainloop()

        except Exception as e:
            messagebox.showerror("Error", f"Screen capture failed:\n{str(e)}")
            traceback.print_exc()
            if 'root' in locals():
                try:
                    root.destroy()
                except:
                    pass

#---------------------------------------------------------------------------------------------------------------------


 #-------------------------------Get Camera and Set Camera--------------------------------------------
    def open_camera(self) -> None:
            """Enhanced cross-platform camera capture with full feature set"""
            try:
                # Check for required imports
                import cv2
                from PIL import Image, ImageTk
                import platform
                from pathlib import Path
                import subprocess
                import time
                import sys

                # Create media_files directory if it doesn't exist
                os.makedirs('media_files', exist_ok=True)

                def get_available_cameras():
                    """Get list of available cameras with proper device detection for all platforms"""
                    available_cameras = []
                    system = platform.system()

                    def check_camera(index, device_path=None, device_name=None):
                        """Helper to check camera capabilities across different platforms"""
                        try:
                            # Initialize camera based on platform
                            if device_path and system == "Linux":
                                cap = cv2.VideoCapture(device_path)
                            elif system == "Windows":
                                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Use DirectShow on Windows
                            else:
                                cap = cv2.VideoCapture(index)

                            if cap.isOpened():
                                ret, _ = cap.read()
                                if ret:
                                    # Store original settings
                                    original_settings = {
                                        cv2.CAP_PROP_FRAME_WIDTH: cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                        cv2.CAP_PROP_FRAME_HEIGHT: cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                        cv2.CAP_PROP_FPS: cap.get(cv2.CAP_PROP_FPS),
                                        cv2.CAP_PROP_BRIGHTNESS: cap.get(cv2.CAP_PROP_BRIGHTNESS),
                                        cv2.CAP_PROP_CONTRAST: cap.get(cv2.CAP_PROP_CONTRAST),
                                        cv2.CAP_PROP_SATURATION: cap.get(cv2.CAP_PROP_SATURATION)
                                    }

                                    # Test resolutions with platform-specific considerations
                                    test_resolutions = [
                                        (3840, 2160),  # 4K
                                        (2560, 1440),  # 2K
                                        (1920, 1080),  # Full HD
                                        (1280, 720),   # HD
                                        (800, 600),    # SVGA
                                        (640, 480)     # VGA
                                    ]

                                    supported_resolutions = []
                                    highest_resolution = None

                                    for width, height in test_resolutions:
                                        try:
                                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                                            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                                            # Verify resolution with actual capture
                                            if actual_w > 0 and actual_h > 0:
                                                test_ret, test_frame = cap.read()
                                                if test_ret and test_frame is not None:
                                                    resolution = f"{actual_w}x{actual_h}"
                                                    if resolution not in supported_resolutions:
                                                        supported_resolutions.append(resolution)
                                                        if not highest_resolution or (actual_w * actual_h > highest_resolution[0] * highest_resolution[1]):
                                                            highest_resolution = (actual_w, actual_h)
                                        except:
                                            continue

                                    # Restore original settings
                                    for prop, value in original_settings.items():
                                        try:
                                            cap.set(prop, value)
                                        except:
                                            pass

                                    # Get current settings after restore
                                    current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    current_fps = int(cap.get(cv2.CAP_PROP_FPS))

                                    # Build camera info dictionary with platform-specific details
                                    camera_info = {
                                        'index': index,
                                        'device_path': device_path,
                                        'name': device_name or f"Camera {index}",
                                        'current_resolution': f"{current_width}x{current_height}",
                                        'supported_resolutions': supported_resolutions,
                                        'highest_resolution': highest_resolution,
                                        'fps': current_fps,
                                        'original_settings': original_settings,
                                        'platform': system
                                    }

                                    cap.release()
                                    return camera_info

                            if cap.isOpened():
                                cap.release()
                        except Exception as e:
                            print(f"Error checking camera {index}: {e}")
                        return None

                    # Platform-specific camera detection
                    if system == "Linux":
                        # Linux: Check /dev/video* devices with udev information
                        try:
                            import glob
                            import pyudev
                            video_devices = sorted(glob.glob('/dev/video*'))
                            context = pyudev.Context()

                            for device_path in video_devices:
                                try:
                                    device = pyudev.Devices.from_device_file(context, device_path)
                                    # Get detailed device information
                                    device_name = device.get('ID_MODEL', '')
                                    vendor = device.get('ID_VENDOR_ID', '')
                                    product = device.get('ID_MODEL_ID', '')

                                    if not device_name:
                                        device_name = f"Camera ({device_path})"
                                    elif vendor and product:
                                        device_name = f"{device_name} ({vendor}:{product})"

                                    index = int(device_path.split('video')[-1])
                                    camera = check_camera(index, device_path, device_name)
                                    if camera:
                                        # Add additional Linux-specific information
                                        camera['device_info'] = {
                                            'vendor': vendor,
                                            'product': product,
                                            'subsystem': device.get('SUBSYSTEM', ''),
                                            'driver': device.get('ID_V4L_DRIVER', '')
                                        }
                                        available_cameras.append(camera)
                                except Exception as e:
                                    print(f"Error processing Linux device {device_path}: {e}")
                                    continue

                        except ImportError:
                            # Fallback if pyudev not available
                            for i in range(10):  # Check first 10 indices
                                camera = check_camera(i)
                                if camera:
                                    available_cameras.append(camera)

                    elif system == "Darwin":  # macOS
                        # Try built-in camera first with specific name
                        camera = check_camera(0, device_name="FaceTime Camera")
                        if camera:
                            # Add macOS-specific information
                            camera['is_builtin'] = True
                            available_cameras.append(camera)

                        # Check additional cameras
                        for i in range(1, 5):
                            camera = check_camera(i)
                            if camera:
                                camera['is_builtin'] = False
                                available_cameras.append(camera)

                    else:  # Windows
                        try:
                            # Try using Windows Management Instrumentation for device info
                            import win32com.client
                            wmi = win32com.client.GetObject("winmgmts:")

                            # Get all video devices from WMI
                            cameras_wmi = {}
                            for device in wmi.InstancesOf("Win32_PnPEntity"):
                                if "Camera" in device.Name or "Webcam" in device.Name:
                                    cameras_wmi[device.DeviceID] = device.Name

                            # Check available cameras with DirectShow
                            for i in range(10):  # Check first 10 indices
                                try:
                                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                                    if cap.isOpened():
                                        # Try to match with WMI device
                                        device_name = next(
                                            (name for id, name in cameras_wmi.items() if str(i) in id),
                                            f"Camera {i}"
                                        )
                                        camera = check_camera(i, device_name=device_name)
                                        if camera:
                                            available_cameras.append(camera)
                                    cap.release()
                                except:
                                    continue

                        except ImportError:
                            # Fallback if WMI access fails
                            for i in range(10):
                                camera = check_camera(i)
                                if camera:
                                    available_cameras.append(camera)

                    return available_cameras

                def start_camera_capture(camera_index, camera_info=None):  # Modified to accept camera_info
                    """Initialize and start camera capture with camera-specific highest resolution"""
                    try:
                        # Create camera dialog
                        dialog = ctk.CTkToplevel(self)
                        dialog.title("Camera Capture")
                        dialog.geometry("800x600")
                        dialog.transient(self)
                        dialog.grab_set()

                        # Center the dialog
                        dialog.update_idletasks()
                        x = (dialog.winfo_screenwidth() - 800) // 2
                        y = (dialog.winfo_screenheight() - 600) // 2
                        dialog.geometry(f"+{x}+{y}")

                        # Initialize camera
                        #cap = cv2.VideoCapture(camera_index)
                        # Initialize camera with default settings
                        if platform.system() == "Linux" and camera_info.get('device_path'):
                            cap = cv2.VideoCapture(camera_info['device_path'])
                        else:
                            cap = cv2.VideoCapture(camera_index)
                        if not cap.isOpened():
                            raise Exception(f"Could not open camera {camera_index}")
                        max_res=None
                        # Only set resolution if highest_resolution is available for this camera
                        if camera_info.get('highest_resolution'):
                            width, height = camera_info['highest_resolution']
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                            # Get actual resolution achieved
                            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                            max_res = (actual_width, actual_height)


                        # Create main content frame
                        content_frame = ctk.CTkFrame(dialog)
                        content_frame.pack(fill="both", expand=True, padx=10, pady=5)

                        # Preview frame
                        preview_frame = ctk.CTkFrame(content_frame)
                        preview_frame.pack(fill="both", expand=True, padx=5, pady=5)

                        preview_label = tk.Label(preview_frame, background="black")
                        preview_label.pack(fill="both", expand=True)

                        # Status frame
                        status_frame = ctk.CTkFrame(dialog)
                        status_frame.pack(fill="x", padx=10, pady=5)

                        # Resolution label
                        if max_res:
                            res_text = f"Resolution: {max_res[0]}x{max_res[1]}"
                        else:
                            res_text = "Resolution: Unknown"

                        resolution_label = ctk.CTkLabel(
                            status_frame,
                            text=res_text,
                            font=("Arial", 12)
                        )
                        resolution_label.pack(side="left", padx=5)

                        # Status label
                        status_label = ctk.CTkLabel(
                            status_frame,
                            text="",
                            font=("Arial", 12)
                        )
                        status_label.pack(side="right", padx=5)

                        # Control frame
                        control_frame = ctk.CTkFrame(dialog)
                        control_frame.pack(fill="x", padx=10, pady=5)

                        # Variables for recording
                        recording_data = {
                            'is_recording': False,
                            'start_time': None,
                            'output': None,
                            'current_file': None
                        }

                        def create_camera_controls(dialog, cap, preview_frame, status_label, recording_data):
                            """Create enhanced camera controls with resolution and file management"""
                            # Create control panels
                            settings_frame = ctk.CTkFrame(dialog)
                            settings_frame.pack(fill="x", padx=10, pady=5)

                            # Resolution controls
                            resolution_frame = ctk.CTkFrame(settings_frame)
                            resolution_frame.pack(side="left", padx=5, pady=5, fill="x", expand=True)

                            resolutions = [
                                "3840x2160 (4K)",
                                "1920x1080 (Full HD)",
                                "1280x720 (HD)",
                                "800x600 (SVGA)",
                                "640x480 (VGA)"
                            ]

                            current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            current_res = f"{current_width}x{current_height}"

                            ctk.CTkLabel(
                                resolution_frame,
                                text="Resolution:",
                                font=("Arial", 12)
                            ).pack(side="left", padx=5)

                            def change_resolution(choice):
                                try:
                                    width, height = map(int, choice.split()[0].split('x'))
                                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                                    if actual_width == width and actual_height == height:
                                        status_label.configure(text=f"Resolution changed to {width}x{height}")
                                    else:
                                        status_label.configure(
                                            text=f"Actual resolution: {actual_width}x{actual_height}"
                                        )
                                    dialog.after(2000, lambda: status_label.configure(text=""))
                                except Exception as e:
                                    status_label.configure(text=f"Error changing resolution: {str(e)}")
                                    dialog.after(2000, lambda: status_label.configure(text=""))

                            resolution_menu = ctk.CTkOptionMenu(
                                resolution_frame,
                                values=resolutions,
                                command=change_resolution,
                                width=150
                            )
                            resolution_menu.pack(side="left", padx=5)

                            # Set current resolution in menu
                            for res in resolutions:
                                if res.startswith(current_res):
                                    resolution_menu.set(res)
                                    break

                            # File management functions
                            def get_save_location(default_name, file_type):
                                """Get save location with custom filename"""
                                file_types = {
                                    'photo': [('PNG files', '*.png'), ('JPEG files', '*.jpg')],
                                    'video': [('MP4 files', '*.mp4'), ('AVI files', '*.avi')]
                                }

                                initialdir = os.path.abspath('media_files')
                                if not os.path.exists(initialdir):
                                    os.makedirs(initialdir)

                                return filedialog.asksaveasfilename(
                                    initialfile=default_name,
                                    defaultextension=file_types[file_type][0][1],
                                    filetypes=file_types[file_type],
                                    initialdir=initialdir,
                                    title=f"Save {file_type.title()} As"
                                )

                            def capture_photo():
                                """Enhanced photo capture with custom save location"""
                                try:
                                    ret, frame = cap.read()
                                    if ret:
                                        # Convert to RGB for PIL
                                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        image = Image.fromarray(rgb_frame)

                                        # Default filename
                                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                                        default_name = f"camera_photo_{timestamp}.png"

                                        # Get save location
                                        filepath = get_save_location(default_name, 'photo')
                                        if not filepath:  # User cancelled
                                            return

                                        # Save image
                                        image.save(filepath)

                                        # Update media entry with relative path if in media_files
                                        rel_path = os.path.relpath(filepath, 'media_files')
                                        if not rel_path.startswith('..'):
                                            self.media_entry.delete(0, 'end')
                                            self.media_entry.insert(0, f"\\file media_files/{rel_path}")
                                        else:
                                            self.media_entry.delete(0, 'end')
                                            self.media_entry.insert(0, f"\\file {filepath}")

                                        # Show success message
                                        status_label.configure(text="✓ Photo saved!")
                                        dialog.after(2000, lambda: status_label.configure(text=""))

                                        # Flash effect
                                        preview_frame.configure(fg_color="white")
                                        dialog.after(100, lambda: preview_frame.configure(fg_color="black"))

                                except Exception as e:
                                    status_label.configure(text=f"Error: {str(e)}")
                                    dialog.after(2000, lambda: status_label.configure(text=""))

                            def toggle_recording():
                                """Enhanced video recording with custom save location"""
                                try:
                                    if not recording_data['is_recording']:
                                        # Get save location first
                                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                                        default_name = f"camera_video_{timestamp}.mp4"
                                        filepath = get_save_location(default_name, 'video')

                                        if not filepath:  # User cancelled
                                            return

                                        # Start recording
                                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                        fps = int(cap.get(cv2.CAP_PROP_FPS))

                                        # Create video writer
                                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                        recording_data['output'] = cv2.VideoWriter(
                                            filepath, fourcc, fps, (width, height)
                                        )
                                        recording_data['current_file'] = filepath
                                        recording_data['is_recording'] = True
                                        recording_data['start_time'] = time.time()

                                        # Update UI
                                        record_button.configure(
                                            text="Stop Recording",
                                            fg_color="#FF4444",
                                            hover_color="#CC3333"
                                        )
                                        photo_button.configure(state="disabled")
                                        resolution_menu.configure(state="disabled")
                                        update_recording_time()

                                    else:
                                        # Stop recording
                                        recording_data['is_recording'] = False
                                        if recording_data['output']:
                                            recording_data['output'].release()

                                        # Update media entry with relative path if in media_files
                                        filepath = recording_data['current_file']
                                        rel_path = os.path.relpath(filepath, 'media_files')
                                        if not rel_path.startswith('..'):
                                            self.media_entry.delete(0, 'end')
                                            self.media_entry.insert(0, f"\\file media_files/{rel_path}")
                                        else:
                                            self.media_entry.delete(0, 'end')
                                            self.media_entry.insert(0, f"\\file {filepath}")

                                        # Reset UI
                                        record_button.configure(
                                            text="Record Video",
                                            fg_color="#4A90E2",
                                            hover_color="#357ABD"
                                        )
                                        photo_button.configure(state="normal")
                                        resolution_menu.configure(state="normal")
                                        status_label.configure(text="✓ Video saved!")
                                        dialog.after(2000, lambda: status_label.configure(text=""))

                                except Exception as e:
                                    status_label.configure(text=f"Error: {str(e)}")
                                    dialog.after(2000, lambda: status_label.configure(text=""))
                                    recording_data['is_recording'] = False
                                    photo_button.configure(state="normal")
                                    resolution_menu.configure(state="normal")

                            def update_recording_time():
                                """Update recording duration display"""
                                if recording_data['is_recording']:
                                    elapsed = time.time() - recording_data['start_time']
                                    minutes = int(elapsed // 60)
                                    seconds = int(elapsed % 60)
                                    status_label.configure(
                                        text=f"Recording: {minutes:02d}:{seconds:02d} ({os.path.basename(recording_data['current_file'])})"
                                    )
                                    dialog.after(1000, update_recording_time)

                            # Create control buttons
                            control_frame = ctk.CTkFrame(dialog)
                            control_frame.pack(fill="x", padx=10, pady=5)

                            photo_button = ctk.CTkButton(
                                control_frame,
                                text="Take Photo",
                                command=capture_photo,
                                width=120,
                                font=("Arial", 13),
                                fg_color="#4A90E2",
                                hover_color="#357ABD"
                            )
                            photo_button.pack(side="left", padx=5)

                            record_button = ctk.CTkButton(
                                control_frame,
                                text="Record Video",
                                command=toggle_recording,
                                width=120,
                                font=("Arial", 13),
                                fg_color="#4A90E2",
                                hover_color="#357ABD"
                            )
                            record_button.pack(side="left", padx=5)

                            close_button = ctk.CTkButton(
                                control_frame,
                                text="Close",
                                command=dialog.destroy,
                                width=100,
                                font=("Arial", 13),
                                fg_color="#FF4444",
                                hover_color="#CC3333"
                            )
                            close_button.pack(side="right", padx=5)

                            return photo_button, record_button, resolution_menu



                        def close_camera():
                            """Clean up and close camera"""
                            try:
                                # Stop recording if active
                                if recording_data['is_recording']:
                                    recording_data['is_recording'] = False
                                    if recording_data['output']:
                                        recording_data['output'].release()

                                # Release camera
                                cap.release()
                                dialog.destroy()

                            except Exception as e:
                                print(f"Error closing camera: {str(e)}")
                                dialog.destroy()

                        # Create controls
                        photo_button, record_button, resolution_menu = create_camera_controls(
                            dialog, cap, preview_frame, status_label, recording_data
                        )

                        close_button = ctk.CTkButton(
                            control_frame,
                            text="Close",
                            command=close_camera,
                            width=100,
                            font=("Arial", 13),
                            fg_color="#FF4444",
                            hover_color="#CC3333"
                        )
                        close_button.pack(side="right", padx=5)

                        def update_preview():
                            """Update camera preview"""
                            if cap.isOpened():
                                ret, frame = cap.read()
                                if ret:
                                    # Record frame if recording
                                    if recording_data['is_recording']:
                                        recording_data['output'].write(frame)

                                    # Convert frame for display
                                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    image = Image.fromarray(rgb_frame)

                                    # Scale for display
                                    display_size = (800, 600)
                                    image.thumbnail(display_size, Image.Resampling.LANCZOS)

                                    # Convert to PhotoImage
                                    photo = ImageTk.PhotoImage(image=image)
                                    preview_label.configure(image=photo)
                                    preview_label.image = photo  # Keep reference

                                    # Schedule next update
                                    dialog.after(10, update_preview)

                        # Start preview
                        update_preview()

                        # Handle window closing
                        dialog.protocol("WM_DELETE_WINDOW", close_camera)

                    except Exception as e:
                        messagebox.showerror("Error", f"Camera initialization failed: {str(e)}")
                        dialog.destroy()

                # Main camera initialization
                cameras = get_available_cameras()
                if not cameras:
                    messagebox.showerror(
                        "No Cameras",
                        "No working cameras found.\nPlease connect a camera and try again."
                    )
                    return

                # Create camera selection dialog if multiple cameras found
                if len(cameras) > 1:
                    select_dialog = ctk.CTkToplevel(self)
                    select_dialog.title("Select Camera")
                    select_dialog.geometry("400x400")
                    select_dialog.transient(self)
                    select_dialog.grab_set()

                    # Center dialog
                    select_dialog.update_idletasks()
                    x = (select_dialog.winfo_screenwidth() - 400) // 2
                    y = (select_dialog.winfo_screenheight() - 400) // 2
                    select_dialog.geometry(f"+{x}+{y}")

                    # Title label
                    ctk.CTkLabel(
                        select_dialog,
                        text="Available Cameras",
                        font=("Arial", 16, "bold")
                    ).pack(pady=10, padx=20)

                    # Create scrollable frame for camera options
                    scroll_frame = ctk.CTkScrollableFrame(
                        select_dialog,
                        width=360,
                        height=280
                    )
                    scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)

                    # Camera selection variable
                    selected_camera = tk.StringVar(value="0")  # Default to first camera

                    # Create radio buttons for each camera
                    for camera in cameras:
                        # Create frame for each camera option
                        camera_frame = ctk.CTkFrame(scroll_frame)
                        camera_frame.pack(fill="x", padx=5, pady=5)

                        # Radio button with camera name
                        rb = ctk.CTkRadioButton(
                            camera_frame,
                            text=camera['name'],
                            variable=selected_camera,
                            value=str(camera['index']),
                            font=("Arial", 13)
                        )
                        rb.pack(side="top", padx=10, pady=2, anchor="w")

                        # Add resolution and FPS info
                        info_text = f"Resolution: {camera['current_resolution']}"
                        if camera.get('supported_resolutions'):
                            info_text += f"\nSupported: {', '.join(camera['supported_resolutions'][:3])}"
                        if camera.get('fps'):
                            info_text += f"\nFPS: {camera['fps']}"

                        info_label = ctk.CTkLabel(
                            camera_frame,
                            text=info_text,
                            font=("Arial", 12),
                            justify="left",
                            text_color="gray"
                        )
                        info_label.pack(side="top", padx=30, pady=2, anchor="w")

                    # Create buttons frame
                    button_frame = ctk.CTkFrame(select_dialog)
                    button_frame.pack(fill="x", padx=10, pady=10)

                    def on_camera_selected():
                        camera_index = int(selected_camera.get())
                        selected_camera_info = next(
                            (cam for cam in cameras if cam['index'] == camera_index),
                            None
                        )
                        select_dialog.destroy()
                        # Start capture with selected camera
                        if selected_camera_info:
                            start_camera_capture(camera_index, selected_camera_info)


                        #start_camera_capture(camera_index)

                    # Select button
                    ctk.CTkButton(
                        button_frame,
                        text="Open Camera",
                        command=on_camera_selected,
                        width=200,
                        font=("Arial", 13),
                        fg_color="#4A90E2",
                        hover_color="#357ABD"
                    ).pack(side="left", padx=10)

                    # Cancel button
                    ctk.CTkButton(
                        button_frame,
                        text="Cancel",
                        command=select_dialog.destroy,
                        width=100,
                        font=("Arial", 13),
                        fg_color="#FF4444",
                        hover_color="#CC3333"
                    ).pack(side="right", padx=10)

                else:
                    # If only one camera, use it directly with its info
                    start_camera_capture(cameras[0]['index'], cameras[0])
                    # If only one camera, use it directly
                    #start_camera_capture(cameras[0]['index'])

            except ImportError as e:
                messagebox.showerror(
                    "Missing Dependencies",
                    "Camera capture requires additional modules.\n" +
                    "Please install required packages:\n\n" +
                    "pip install opencv-python pillow"
                )
            except Exception as e:
                messagebox.showerror(
                    "Camera Error",
                    f"Camera initialization failed:\n{str(e)}\n\n" +
                    "Please check your camera connection and permissions."
                )


 #----------------------------------------------------------------------------


    def update_notes_buttons(self, active_mode: str) -> None:
        """Update button colors based on active mode"""
        for mode, btn_info in self.notes_buttons.items():
            if mode == active_mode:
                btn_info['button'].configure(
                    fg_color=btn_info['active_color'],
                    hover_color=btn_info['hover_color']
                )
            else:
                btn_info['button'].configure(
                    fg_color="gray",
                    hover_color="#4A4A4A"
                )


    def create_toolbar(self) -> None:
        """Create main editor toolbar with presentation features and capture controls split into two rows"""
        # Create main toolbar container
        self.toolbar = ctk.CTkFrame(self)
        self.toolbar.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        # Upper row for file and presentation operations
        upper_row = ctk.CTkFrame(self.toolbar)
        upper_row.pack(fill="x", padx=5, pady=(5, 2))

        # Basic file operations buttons
        buttons_upper = [
            ("New", self.new_file, "Create new presentation"),
            ("Open", self.open_file, "Open existing presentation"),
            ("Save", self.save_file, "Save current presentation"),
            ("Convert to TeX", self.convert_to_tex, "Convert to LaTeX format"),
            ("Generate PDF", self.generate_pdf, "Generate PDF file"),
            ("Present with Notes", self.present_with_notes, "Launch dual-screen presentation with notes"),
            ("Preview PDF", self.preview_pdf, "View generated PDF"),
            ("Load TeX", self.load_tex_file, "Load and convert Beamer TeX file"),
            ("Overwrite TeX+PDF", self.overwrite_tex_and_generate_pdf, "Convert back to TeX and generate PDF")
        ]

        for text, command, tooltip in buttons_upper:
            if text == "Export to Overleaf":
                btn = ctk.CTkButton(
                    upper_row,
                    text=text,
                    command=command,
                    width=120,
                    fg_color="#47A141",
                    hover_color="#2E8B57"
                )
            elif text == "Present with Notes":
                btn = ctk.CTkButton(
                    upper_row,
                    text=text,
                    command=command,
                    width=120,
                    fg_color="#4A90E2",
                    hover_color="#357ABD"
                )
            else:
                btn = ctk.CTkButton(
                    upper_row,
                    text=text,
                    command=command,
                    width=100
                )
            btn.pack(side="left", padx=5)
            self.create_tooltip(btn, tooltip)

        # Lower row for screen capture controls
        lower_row = ctk.CTkFrame(self.toolbar)
        lower_row.pack(fill="x", padx=5, pady=(2, 5))

        # Left side - Screen capture controls
        capture_frame = ctk.CTkFrame(lower_row, fg_color="transparent")
        capture_frame.pack(side="left", padx=5)

        # Screen capture label
        capture_label = ctk.CTkLabel(capture_frame, text="Screen Capture:")
        capture_label.pack(side="left", padx=5)
        self.create_tooltip(capture_label, "Choose capture mode and settings")

        # Initialize capture settings
        self.capture_mode = tk.StringVar(value="single")
        self.frame_count = tk.IntVar(value=10)
        self.frame_delay = tk.DoubleVar(value=0.5)

        # Single frame mode
        single_btn = ctk.CTkRadioButton(
            capture_frame,
            text="Single",
            variable=self.capture_mode,
            value="single"
        )
        single_btn.pack(side="left", padx=5)
        self.create_tooltip(single_btn, "Capture single screenshot")

        # Animation mode
        anim_btn = ctk.CTkRadioButton(
            capture_frame,
            text="Animation",
            variable=self.capture_mode,
            value="animation"
        )
        anim_btn.pack(side="left", padx=5)
        self.create_tooltip(anim_btn, "Capture animated GIF")

        # Animation settings (shown/hidden based on mode)
        self.anim_settings = ctk.CTkFrame(capture_frame, fg_color="transparent")

        # Frames control
        frames_frame = ctk.CTkFrame(self.anim_settings, fg_color="transparent")
        frames_frame.pack(side="left", padx=5)
        ctk.CTkLabel(frames_frame, text="Frames:").pack(side="left")
        frames_entry = ctk.CTkEntry(frames_frame, textvariable=self.frame_count, width=40)
        frames_entry.pack(side="left", padx=2)
        self.create_tooltip(frames_entry, "Number of frames to capture")

        # Delay control
        delay_frame = ctk.CTkFrame(self.anim_settings, fg_color="transparent")
        delay_frame.pack(side="left", padx=5)
        ctk.CTkLabel(delay_frame, text="Delay:").pack(side="left")
        delay_entry = ctk.CTkEntry(delay_frame, textvariable=self.frame_delay, width=40)
        delay_entry.pack(side="left", padx=2)
        self.create_tooltip(delay_entry, "Delay between frames (seconds)")

        # Capture button
        capture_btn = ctk.CTkButton(
            capture_frame,
            text="Capture",
            command=self.capture_screen,
            width=80,
            fg_color="#4A90E2",
            hover_color="#357ABD"
        )
        capture_btn.pack(side="left", padx=5)
        self.create_tooltip(capture_btn, "Start screen capture")

        # Add separator
        ttk.Separator(lower_row, orient="vertical").pack(side="left", padx=10, pady=5, fill="y")

        # Right side - Moved buttons from top menu
        right_buttons = [
            ("Edit Preamble", self.edit_preamble, "Edit LaTeX preamble"),
            ("Presentation Settings", self.show_settings_dialog, "Configure presentation settings"),
            ("Get Source", self.get_source_from_tex, "Extract source from TEX file"),
            ("Export to Overleaf", self.create_overleaf_zip, "Create Overleaf-compatible zip")
        ]

        for text, command, tooltip in right_buttons:
            btn = ctk.CTkButton(
                lower_row,
                text=text,
                command=command,
                width=130
            )
            btn.pack(side="left", padx=5)
            self.create_tooltip(btn, tooltip)

        # Function to show/hide animation settings
        def toggle_anim_settings(*args):
            if self.capture_mode.get() == "animation":
                self.anim_settings.pack(side='left', padx=5)
            else:
                self.anim_settings.pack_forget()

        # Bind mode changes
        self.capture_mode.trace('w', toggle_anim_settings)

        # Initial state
        toggle_anim_settings()
#------------------------------------------------------------------------------------------------------

    def on_notes_mode_change(self, mode: str) -> None:
        """Handle notes mode change"""
        self.notes_mode.set(mode)

        # Update button colors
        for btn in self.mode_buttons:
            if btn.mode == mode:
                btn.configure(fg_color=btn.active_color)
            else:
                btn.configure(fg_color="gray")

        # Configure editor state
        if mode == "slides":
            self.notes_editor.configure(state="disabled")
        else:
            self.notes_editor.configure(state="normal")



    def create_tooltip(self, widget, text):
        """Create tooltip for widget"""
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")

            label = tk.Label(tooltip, text=text, justify='left',
                           background="#ffffe0", relief='solid', borderwidth=1,
                           font=("Arial", 10))
            label.pack()

            def hide_tooltip():
                tooltip.destroy()

            widget.tooltip = tooltip
            widget.tooltip_timer = self.after(2000, hide_tooltip)

        def hide_tooltip(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                if hasattr(widget, 'tooltip_timer'):
                    self.after_cancel(widget.tooltip_timer)

        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)

    def generate_odp(self) -> None:
        """Generate ODP presentation with automatic TEX generation if needed"""
        if not self.current_file:
            messagebox.showwarning("Warning", "Please save your file first!")
            return

        try:
            #self.save_file()  # Save current state to ensure latest content

            # Get base filename without extension
            base_filename = os.path.splitext(self.current_file)[0]
            tex_file = base_filename + '.tex'

            # Clear terminal
            self.clear_terminal()
            self.write_to_terminal("Starting ODP generation process...\n")

            # Check if TEX file exists and generate if needed
            if not os.path.exists(tex_file):
                self.write_to_terminal("TEX file not found. Generating from source...\n")
                try:
                    from BeamerSlideGenerator import process_input_file
                    process_input_file(self.current_file, tex_file)
                    self.write_to_terminal("✓ TEX file generated successfully\n", "green")
                except Exception as e:
                    self.write_to_terminal(f"✗ Error generating TEX file: {str(e)}\n", "red")
                    raise Exception("TEX generation failed")

            # Convert TEX to ODP
            self.write_to_terminal("Converting TEX to ODP...\n")
            try:
                #from Beam2odp import BeamerToODP
                converter = BeamerToODP(tex_file)
                self.write_to_terminal("Parsing TEX content...\n")
                converter.parse_input()

                self.write_to_terminal("Generating ODP file...\n")
                odp_file = converter.generate_odp()

                if odp_file and os.path.exists(odp_file):
                    self.write_to_terminal("✓ ODP file generated successfully!\n", "green")

                    # Ask to open the generated file
                    if messagebox.askyesno("Success",
                                         "ODP presentation generated successfully! Would you like to open it?"):
                        if sys.platform.startswith('win'):
                            os.startfile(odp_file)
                        elif sys.platform.startswith('darwin'):
                            subprocess.run(['open', odp_file])
                        else:
                            subprocess.run(['xdg-open', odp_file])
                else:
                    self.write_to_terminal("✗ Error: No output file was generated\n", "red")

            except Exception as e:
                error_text = f"✗ Error in ODP conversion: {str(e)}\n"
                error_text += "Detailed error information:\n"
                error_text += traceback.format_exc()
                self.write_to_terminal(error_text, "red")
                raise Exception("ODP conversion failed")

        except Exception as e:
            messagebox.showerror("Error", f"Error generating ODP presentation:\n{str(e)}")
            print(f"Error details: {str(e)}")
            traceback.print_exc()

    def get_required_media_files(self, tex_content: str) -> set:
        """Parse TEX file to identify all required media files including multimedia content and previews"""
        required_files = set()

        # Regular expressions for different media references
        patterns = {
            'images': [
                r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}',    # Standard images
                r'\\pgfimage(?:\[.*?\])?\{([^}]+)\}',          # PGF images
                r'media_files/([^}]+_preview\.png)'             # Preview images
            ],
            'video': [
                r'\\movie(?:\[.*?\])?\{.*?\}\{\.?/?media_files/([^}]+)\}',  # Movie elements (handle ./ prefix)
                r'\\href\{run:([^}]+)\}',                       # Runnable media links
                r'\\movie\[.*?\]\{.*?\}\{([^}]+)\}'            # Movie with options
            ],
            'animations': [
                r'\\animategraphics(?:\[.*?\])?\{[^}]*\}\{([^}]+)\}',  # Animated graphics
                r'\\animate(?:\[.*?\])?\{[^}]*\}\{([^}]+)\}'           # General animations
            ],
            'audio': [
                r'\\sound(?:\[.*?\])?\{.*?\}\{([^}]+)\}',      # Sound elements
                r'\\audiofile\{([^}]+)\}'                       # Audio files
            ],
            'general_media': [
                r'\\file\s+media_files/([^\s}]+)',             # General media files
                r'\\play\s+\\file\s+media_files/([^\s}]+)',    # Playable media
                r'\\mediapath\{([^}]+)\}'                       # Media path references
            ]
        }

        self.write_to_terminal("\nAnalyzing required media files:\n")

        # Find all media references
        for media_type, pattern_list in patterns.items():
            self.write_to_terminal(f"\nChecking {media_type} references:\n")
            for pattern in pattern_list:
                matches = re.finditer(pattern, tex_content)
                for match in matches:
                    filepath = match.group(1)
                    # Clean up the path
                    filepath = filepath.replace('media_files/', '')
                    filepath = filepath.replace('./', '')  # Remove any ./ prefix
                    filepath = filepath.strip()

                    # Add the file to required files
                    required_files.add(filepath)
                    self.write_to_terminal(f"  ✓ Found: {filepath}\n", "green")

                    # If this is a preview image, also add the corresponding video
                    if filepath.endswith('_preview.png'):
                        base_video_name = filepath.replace('_preview.png', '.mp4')
                        required_files.add(base_video_name)
                        self.write_to_terminal(f"  ✓ Added corresponding video: {base_video_name}\n", "green")

                    # If this is a video, check for its preview image
                    if filepath.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        preview_name = filepath.rsplit('.', 1)[0] + '_preview.png'
                        if preview_name not in required_files:
                            required_files.add(preview_name)
                            self.write_to_terminal(f"  ✓ Added corresponding preview: {preview_name}\n", "green")

        return required_files

    def verify_media_files(self, required_files: set) -> tuple:
        """Verify existence of required media files and classify them"""
        verified_files = set()
        missing_files = set()
        media_types = {
            'images': [],
            'videos': [],
            'audio': [],
            'animations': [],
            'other': []
        }

        for filepath in required_files:
            full_path = os.path.join('media_files', filepath)
            if os.path.exists(full_path):
                verified_files.add(filepath)
                # Classify file by extension
                ext = os.path.splitext(filepath)[1].lower()
                if ext in ['.png', '.jpg', '.jpeg', '.pdf', '.eps']:
                    media_types['images'].append(filepath)
                elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
                    media_types['videos'].append(filepath)
                elif ext in ['.mp3', '.wav', '.ogg', '.m4a', '.flac']:
                    media_types['audio'].append(filepath)
                elif ext in ['.gif', '.webp']:
                    media_types['animations'].append(filepath)
                else:
                    media_types['other'].append(filepath)
            else:
                missing_files.add(filepath)

        return verified_files, missing_files, media_types

    def create_manifest(self, tex_file: str, verified_files: set, missing_files: set, media_types: dict) -> str:
        """Create detailed manifest content"""
        manifest_content = [
            "# Project Media Files Manifest",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Project Files:",
            f"- {os.path.basename(tex_file)} (Main TeX file)",
            "\n## Media Files by Type:"
        ]

        # Add categorized media files
        for media_type, files in media_types.items():
            if files:
                manifest_content.extend([
                    f"\n### {media_type.title()}:",
                    *[f"- media_files/{file}" for file in sorted(files)]
                ])

        # Add missing files section if any
        if missing_files:
            manifest_content.extend([
                "\n## Missing Files (Please Check):",
                *[f"- {file}" for file in sorted(missing_files)]
            ])

        manifest_content.extend([
            "\n## File Statistics:",
            f"Total Files: {len(verified_files) + len(missing_files)}",
            f"Successfully Included: {len(verified_files)}",
            f"Missing: {len(missing_files)}"
        ])

        return '\n'.join(manifest_content)

    def create_overleaf_zip(self) -> None:
        """Create a zip file compatible with Overleaf containing tex and all required media files"""
        if not self.current_file:
            messagebox.showwarning("Warning", "Please save your file first!")
            return

        try:
            # First ensure current state is saved and tex is generated
            #self.save_file()

            # Get base filename without extension
            base_filename = os.path.splitext(self.current_file)[0]
            tex_file = base_filename + '.tex'

            # Clear terminal and show progress
            self.clear_terminal()
            self.write_to_terminal("Creating Overleaf-compatible zip file...\n")

            # Convert to tex if not already done
            if not os.path.exists(tex_file):
                self.write_to_terminal("Generating TeX file...\n")
                from BeamerSlideGenerator import process_input_file
                process_input_file(self.current_file, tex_file)
                self.write_to_terminal("✓ TeX file generated successfully\n", "green")

            # Read TEX content and identify required files
            with open(tex_file, 'r', encoding='utf-8') as f:
                tex_content = f.read()

            # Create the progress dialog
            self.update_idletasks()  # Ensure main window is updated
            progress = self.create_progress_dialog(
                "Creating Zip File",
                "Analyzing required files..."
            )

            try:
                # Analyze and verify media files
                required_files = self.get_required_media_files(tex_content)
                verified_files, missing_files, media_types = self.verify_media_files(required_files)

                # Create zip file
                zip_filename = base_filename + '_overleaf.zip'
                total_files = len(verified_files) + 1  # +1 for tex file
                processed_files = 0

                progress.update_progress(0, "Creating zip file...")

                with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add the tex file
                    progress.update_progress(
                        (processed_files / total_files) * 100,
                        "Adding TeX file..."
                    )
                    zipf.write(tex_file, os.path.basename(tex_file))
                    processed_files += 1
                    self.write_to_terminal(f"✓ Added: {os.path.basename(tex_file)}\n", "green")

                    # Add verified media files
                    self.write_to_terminal("\nAdding media files:\n")
                    for filename in verified_files:
                        file_path = os.path.join('media_files', filename)
                        progress.update_progress(
                            (processed_files / total_files) * 100,
                            f"Adding {filename}..."
                        )

                        # Ensure the media_files directory exists in the zip
                        if processed_files == 1:  # First media file
                            zipf.writestr('media_files/.keep', '')  # Create empty file to ensure directory exists

                        zipf.write(file_path, os.path.join('media_files', filename))
                        self.write_to_terminal(f"✓ Added: {filename}\n", "green")
                        processed_files += 1

                    # Create detailed manifest
                    manifest_content = self.create_manifest(
                        tex_file, verified_files, missing_files, media_types
                    )
                    zipf.writestr('manifest.txt', manifest_content)

                progress.update_progress(100, "Complete!")
                time.sleep(0.5)  # Brief pause to show completion

                # Show completion message with details
                message = [f"Zip file created successfully!"]

                # Add statistics
                stats = []
                total_files = sum(len(files) for files in media_types.values())
                if total_files > 0:
                    stats.extend([
                        "",
                        "Media files included:",
                        *[f"- {media_type.title()}: {len(files)} files"
                          for media_type, files in media_types.items() if files]
                    ])

                # Add warnings for missing files
                if missing_files:
                    stats.extend([
                        "",
                        f"Warning: {len(missing_files)} required files were missing.",
                        "Check manifest.txt in the zip file for details."
                    ])

                # Add total size
                try:
                    zip_size = os.path.getsize(zip_filename)
                    stats.append("")
                    stats.append(f"Total zip size: {self.format_file_size(zip_size)}")
                except OSError:
                    pass

                message.extend(stats)
                message.append("\nWould you like to open the containing folder?")

                # Close progress dialog before showing message
                progress.close()

                if messagebox.askyesno("Success", "\n".join(message)):
                    # Open the folder containing the zip file
                    if sys.platform.startswith('win'):
                        os.system(f'explorer /select,"{zip_filename}"')
                    elif sys.platform.startswith('darwin'):
                        subprocess.run(['open', '-R', zip_filename])
                    else:
                        subprocess.run(['xdg-open', os.path.dirname(zip_filename)])

            except Exception as e:
                if 'progress' in locals():
                    progress.close()
                raise

        except Exception as e:
            error_msg = f"Error creating zip file: {str(e)}\n"
            self.write_to_terminal(f"✗ {error_msg}", "red")
            traceback.print_exc()  # Print full traceback to help with debugging
            messagebox.showerror("Error", f"Error creating zip file:\n{str(e)}")

    def format_file_size(self, size: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def create_progress_dialog(self, title: str, message: str) -> 'ProgressDialog':
        """Create a progress dialog window"""
        class ProgressDialog:
            def __init__(self, parent, title, message):
                self.window = ctk.CTkToplevel(parent)
                self.window.title(title)
                self.window.geometry("300x150")
                self.window.transient(parent)

                # Center the window
                self.window.update_idletasks()
                width = self.window.winfo_width()
                height = self.window.winfo_height()
                x = (self.window.winfo_screenwidth() // 2) - (width // 2)
                y = (self.window.winfo_screenheight() // 2) - (height // 2)
                self.window.geometry(f'+{x}+{y}')

                self.message = ctk.CTkLabel(self.window, text=message)
                self.message.pack(pady=10)

                self.progress = ctk.CTkProgressBar(self.window)
                self.progress.pack(pady=10, padx=20, fill="x")
                self.progress.set(0)

                self.progress_text = ctk.CTkLabel(self.window, text="0%")
                self.progress_text.pack(pady=5)

                # Wait for window to be visible before grabbing
                self.window.wait_visibility()
                self.window.grab_set()

                # Keep dialog on top
                self.window.lift()
                self.window.focus_force()

            def update_progress(self, value: float, message: str = None) -> None:
                """Update progress bar and message"""
                try:
                    if self.window.winfo_exists():
                        self.progress.set(value / 100)
                        self.progress_text.configure(text=f"{value:.1f}%")
                        if message:
                            self.message.configure(text=message)
                        self.window.update()
                except tk.TclError:
                    pass  # Window might have been closed

            def close(self) -> None:
                """Close the progress dialog"""
                try:
                    if self.window.winfo_exists():
                        self.window.grab_release()
                        self.window.destroy()
                except tk.TclError:
                    pass  # Window might have been closed

        return ProgressDialog(self, title, message)



    def open_presentation(self, file_path):
        """Open the generated presentation with appropriate application"""
        try:
            if sys.platform.startswith('win'):
                os.startfile(file_path)
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', file_path])
            else:
                subprocess.run(['xdg-open', file_path])
        except Exception as e:
            self.write_to_terminal(f"Error opening presentation: {str(e)}\n", "red")
            messagebox.showerror("Error",
                               f"Error opening presentation:\n{str(e)}")

    def create_context_menu(self) -> None:
        """Create right-click context menu"""
        self.context_menu = ctk.CTkFrame(self)

        commands = [
            ("Add Bullet", "- "),
            ("Add textcolor", "\\textcolor[RGB]{255,165,0}{}"),
            ("Add Media Directive", "\\file media_files/"),
            ("Add Play Directive", "\\play "),
            ("Add URL", "https://"),
            ("Add Comment", "% ")
        ]

        for text, insert_text in commands:
            ctk.CTkButton(
                self.context_menu,
                text=text,
                command=lambda t=insert_text: self.insert_text(t)
            ).pack(fill="x", padx=2, pady=2)


    def toggle_highlighting(self) -> None:
        """Toggle syntax highlighting"""
        self.syntax_highlighter.toggle()
        # Also toggle spell checking if enabled
        if self.spell_checking_enabled:
            if self.syntax_highlighter.active:
                self.check_spelling(None)
            else:
                # Clear all misspelled tags if highlighting is disabled
                for widget in [self.content_editor._textbox, self.notes_editor._textbox]:
                    widget.tag_remove("misspelled", "1.0", "end")

    def show_context_menu(self, event) -> None:
        """Show context menu at mouse position"""
        self.context_menu.place(x=event.x_root, y=event.y_root)

    def hide_context_menu(self, event) -> None:
        """Hide context menu"""
        self.context_menu.place_forget()

    def insert_text(self, text: str) -> None:
        """Insert text at cursor position"""
        self.content_editor.insert("insert", text)
        self.hide_context_menu(None)
        if self.syntax_highlighter.active:
            self.syntax_highlighter.highlight()

    # File Operations
    def new_file(self) -> None:
        """Create new presentation"""
        self.current_file = None
        self.slides = []
        self.current_slide_index = -1
        self.update_slide_list()
        self.clear_editor()

        # Reset presentation info
        self.presentation_info = {
            'title': '',
            'subtitle': '',
            'author': '',
            'institution': '',
            'short_institute': '',
            'date': '\\today'
        }

    def open_file(self) -> None:
        global working_folder
        global original_dir
        """Open existing presentation"""
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.load_file(filename)
            # Change to tet file directory
            working_folder= os.path.dirname(filename) or '.'
            os.chdir(working_folder)
            original_dir=working_folder
            # Update working directory in terminal
            self.terminal.set_working_directory(working_folder)

#-----------------------------------------------------------------------------
    def save_file(self) -> None:
        """Save presentation preserving custom preamble and line-level masking"""
        if not self.current_file:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                self.current_file = filename
                global working_folder
                # Change to text file directory
                working_folder = os.path.dirname(filename) or '.'
                os.chdir(working_folder)

                # Update working directory in terminal
                self.terminal.set_working_directory(working_folder)
            else:
                return

        # Save current slide before generating content
        self.save_current_slide()

        try:
            # Get custom preamble with logo
            content = self.get_custom_preamble()

            # Track counts
            masked_count = 0
            total_hidden_lines = 0

            # Add slides in BeamerSlideGenerator's expected format
            for idx, slide in enumerate(self.slides):
                # Check if this slide is masked (deleted)
                is_fully_masked = slide.get('_fully_masked', False) or self.is_slide_masked(idx)

                # Get hidden line indices
                hidden_content_indices = set(slide.get('_hidden_content_indices', []))
                hidden_note_indices = set(slide.get('_hidden_note_indices', []))

                if is_fully_masked:
                    masked_count += 1
                    # Add masked slide marker
                    content += "\n\n% ========== MASKED SLIDE (DELETED) ==========\n"

                content += "\n\n"  # Add two extra line break before new slide

                # Handle title - comment if fully masked
                if is_fully_masked:
                    content += f"% \\title {slide['title']}\n"
                    content += "% \\begin{Content}\n"
                else:
                    content += f"\\title {slide['title']}\n"
                    content += "\\begin{Content}\n"

                # Handle media/TikZ content with masking preservation
                media = slide.get('media', '')
                if media and media != "\\None":
                    # Check if media is masked (starts with % in the stored value)
                    is_media_masked = media.startswith('%') if media else False

                    if is_fully_masked or is_media_masked:
                        content += "% "
                        # Check if it's TikZ code (multi-line)
                        if "\\begin{tikzpicture}" in media:
                            tikz_lines = media.split('\n')
                            for i, line in enumerate(tikz_lines):
                                # Remove any existing % prefix for consistent masking
                                clean_line = line.lstrip('%').lstrip()
                                if i == 0:
                                    content += f"% {clean_line}\n"
                                else:
                                    content += f"% {clean_line}\n"
                        else:
                            # Remove any existing % prefix
                            clean_media = media.lstrip('%').lstrip()
                            content += f"% {clean_media}\n"
                    else:
                        content += "\n"
                        # Check if it's TikZ code
                        if "\\begin{tikzpicture}" in media:
                            # For TikZ, write the complete code
                            if "% TikZ Diagram:" in media:
                                tikz_lines = media.split('\n')[1:]
                                for line in tikz_lines:
                                    content += f"{line}\n"
                            else:
                                content += f"{media}\n"
                        else:
                            content += f"{media}\n"

                # Format content items - PRESERVE LINE-LEVEL MASKING
                content_items = slide.get('content', [])
                if content_items:
                    if not is_fully_masked:
                        content += "\n"  # Add separation between media and content

                for i, item in enumerate(content_items):
                    if item and item.strip():
                        # Check if this specific line should be masked
                        is_line_hidden = i in hidden_content_indices

                        if is_fully_masked or is_line_hidden:
                            # Ensure the line starts with % for masking
                            if not item.strip().startswith('%'):
                                content += f"% {item}\n"
                                total_hidden_lines += 1
                            else:
                                content += f"{item}\n"
                        else:
                            content += f"{item}\n"

                # End Content block
                if is_fully_masked:
                    content += "% \\end{Content}\n\n"
                else:
                    content += "\\end{Content}\n\n"

                # Handle notes with line-level masking preservation
                if is_fully_masked:
                    content += "% \\begin{Notes}\n"
                    if 'notes' in slide and slide['notes']:
                        for i, note in enumerate(slide['notes']):
                            if note and note.strip():
                                is_note_hidden = i in hidden_note_indices
                                if is_note_hidden:
                                    if not note.strip().startswith('%'):
                                        content += f"% {note}\n"
                                        total_hidden_lines += 1
                                    else:
                                        content += f"{note}\n"
                                else:
                                    # Even for visible notes in masked slide, add comment
                                    content += f"% {note}\n"
                    content += "% \\end{Notes}\n"
                else:
                    if 'notes' in slide and slide['notes']:
                        content += "\\begin{Notes}\n"
                        for i, note in enumerate(slide['notes']):
                            if note and note.strip():
                                is_note_hidden = i in hidden_note_indices
                                if is_note_hidden:
                                    # Add % prefix for hidden notes
                                    content += f"% {note}\n"
                                    total_hidden_lines += 1
                                else:
                                    content += f"{note}\n"
                        content += "\\end{Notes}\n"
                    else:
                        content += "\\begin{Notes}\n"
                        content += "\n"
                        content += "\\end{Notes}\n"

            content += "\\end{document}"

            # Save to text file
            with open(self.current_file, 'w', encoding='utf-8') as f:
                f.write(content)

            self.write("✓ File saved successfully: " + self.current_file + "\n", "green")

            # Show summary of masked content
            summary_parts = []
            if masked_count > 0:
                summary_parts.append(f"{masked_count} fully masked slide(s)")
            if total_hidden_lines > 0:
                summary_parts.append(f"{total_hidden_lines} hidden line(s)")

            if summary_parts:
                self.write(f"ℹ Saved with: {', '.join(summary_parts)}\n", "yellow")

        except Exception as e:
            self.write(f"✗ Error saving file: {str(e)}\n", "red")
            messagebox.showerror("Error", f"Error saving file:\n{str(e)}")

    def convert_media_to_latex(self, line: str) -> str:
        """Convert media directives to proper LaTeX commands"""
        if '\\file' in line:
            # Convert \file to \includegraphics
            media_path = line.split('\\file')[-1].strip()
            return f"\\includegraphics[width=0.48\\textwidth,keepaspectratio]{{{media_path}}}"
        elif '\\play' in line:
            # Handle play directives
            if '\\file' in line:
                media_path = line.split('\\file')[-1].strip()
                # Using raw string to handle nested braces
                return (r"\movie[externalviewer]{"
                       r"\includegraphics[width=0.48\textwidth,keepaspectratio]"
                       f"{{{media_path}}}}}{{{media_path}}}")
            elif '\\url' in line:
                url = line.split('\\url')[-1].strip()
                return f"\\href{{{url}}}{{\\textcolor{{blue}}{{Click to play video}}}}"
        elif '\\None' in line:
            return ""  # Return empty string for \None directive
        return line

    def convert_to_tex(self):
        """Convert text to TeX while preserving line-level masking"""
        if not self.current_file:
            messagebox.showwarning("Warning", "Please save your file first!")
            return

        try:
            self.save_file()
            base_filename = os.path.splitext(self.current_file)[0]
            tex_file = base_filename + '.tex'

            self.clear_terminal()
            self.write("Converting text to TeX while preserving masked content...\n")

            # Generate TeX content with preserved masking
            tex_content = self.generate_tex_with_preserved_masking()

            if tex_content:
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(tex_content)
                self.write(f"✓ TeX file generated: {tex_file}\n", "green")

                # Show summary of masked content in TeX
                self.write_tex_masking_summary(tex_content)
            else:
                self.write("✗ Failed to generate TeX content\n", "red")

        except Exception as e:
            self.write(f"✗ Error in conversion: {str(e)}\n", "red")
            messagebox.showerror("Error", f"Error converting to TeX:\n{str(e)}")

    def generate_tex_with_preserved_masking(self):
        """Generate TeX content - remove masked content, preserve original styling"""
        try:
            # Get the original custom preamble (with all styling)
            tex_content = self.get_custom_preamble()

            # Find where \begin{document} is
            doc_pos = tex_content.find("\\begin{document}")
            if doc_pos == -1:
                self.write("Error: Could not find \\begin{document} in preamble\n", "red")
                return None

            # Split into preamble and document body
            preamble = tex_content[:doc_pos]

            # Start building new document body
            new_document_body = "\\begin{document}\n\n"

            # Create title page slide
            new_document_body += "\\begin{frame}[plain]\n"
            new_document_body += "\\titlepage\n"
            new_document_body += "\\end{frame}\n\n"

            # Track statistics
            total_visible_slides = 0
            total_skipped_slides = 0

            # Process each slide
            for idx, slide in enumerate(self.slides):
                is_fully_masked = slide.get('_fully_masked', False)

                if is_fully_masked:
                    total_skipped_slides += 1
                    continue

                hidden_content_indices = set(slide.get('_hidden_content_indices', []))
                hidden_note_indices = set(slide.get('_hidden_note_indices', []))
                media_masked = slide.get('_media_masked', False)

                media = slide.get('media', '')
                if media_masked:
                    media = ""

                # Check if media is a valid image file
                has_valid_image = False
                image_path = None

                if media and media != "\\None" and not media_masked:
                    if media.startswith('\\file'):
                        file_path = media.replace('\\file', '').strip()
                        if file_path and 'example-image' not in file_path.lower():
                            if os.path.exists(file_path):
                                has_valid_image = True
                                image_path = file_path
                            elif os.path.exists(f"media_files/{file_path}"):
                                has_valid_image = True
                                image_path = f"media_files/{file_path}"

                visible_content = []
                for i, line in enumerate(slide.get('content', [])):
                    if line.strip() and i not in hidden_content_indices:
                        visible_content.append(line)

                visible_notes = []
                for i, note in enumerate(slide.get('notes', [])):
                    if note.strip() and i not in hidden_note_indices:
                        visible_notes.append(note)

                if not visible_content and not has_valid_image and not visible_notes:
                    total_skipped_slides += 1
                    continue

                total_visible_slides += 1

                slide_title = slide.get('title', 'Untitled')
                clean_title = re.sub(r'^\[DELETED\]\s*', '', slide_title)

                new_document_body += f"\\begin{{frame}}{{{clean_title}}}\n"
                new_document_body += f"\\frametitle{{{clean_title}}}\n\n"

                # Only use two-column layout if we have a valid image
                if has_valid_image:
                    new_document_body += "\\begin{columns}[T]\n"
                    new_document_body += "\\column{0.45\\textwidth}\n"
                    new_document_body += f"\\begin{{center}}\n"
                    new_document_body += f"\\includegraphics[width=\\textwidth,keepaspectratio]{{{image_path}}}\n"
                    new_document_body += f"\\end{{center}}\n\n"
                    new_document_body += "\\column{0.5\\textwidth}\n"
                    content_col = new_document_body
                else:
                    content_col = new_document_body

                # Process content with proper environment handling
                i = 0
                in_list = False
                in_block = False
                list_type = None
                skip_next_end = False

                while i < len(visible_content):
                    line = visible_content[i].rstrip()
                    if not line.strip():
                        if in_list:
                            content_col += f"\\end{{{list_type}}}\n\n"
                            in_list = False
                            list_type = None
                        if in_block:
                            content_col += "\\end{block}\n\n"
                            in_block = False
                        i += 1
                        continue

                    # Handle existing itemize/enumerate environments
                    if '\\begin{itemize}' in line or '\\begin{enumerate}' in line:
                        if in_list:
                            content_col += f"\\end{{{list_type}}}\n\n"
                        list_type = 'itemize' if 'itemize' in line else 'enumerate'
                        content_col += f"\\begin{{{list_type}}}\n"
                        in_list = True
                        i += 1
                        continue

                    elif '\\end{itemize}' in line or '\\end{enumerate}' in line:
                        if in_list:
                            content_col += f"\\end{{{list_type}}}\n\n"
                            in_list = False
                            list_type = None
                        i += 1
                        continue

                    # Handle block environments
                    if '\\begin{block}' in line:
                        if in_block:
                            content_col += "\\end{block}\n\n"
                        block_match = re.search(r'\\begin{block}\{([^}]*)\}', line)
                        if block_match:
                            block_title = block_match.group(1)
                            content_col += f"\\begin{{block}}{{{block_title}}}\n"
                        else:
                            content_col += "\\begin{block}{}\n"
                        in_block = True
                        i += 1
                        continue

                    elif '\\end{block}' in line:
                        if in_block:
                            content_col += "\\end{block}\n\n"
                            in_block = False
                        i += 1
                        continue

                    # Handle columns environments - pass through as-is
                    if '\\begin{columns}' in line or '\\end{columns}' in line or '\\column' in line:
                        content_col += f"{line}\n"
                        i += 1
                        continue

                    # Handle item commands or bullet points
                    if '\\item' in line:
                        if not in_list:
                            list_type = 'itemize'
                            content_col += f"\\begin{{{list_type}}}\n"
                            in_list = True
                        content_col += f"{line}\n"
                        i += 1
                        continue

                    # Handle bullet points (lines starting with - or •)
                    if line.strip().startswith(('- ', '• ')):
                        if not in_list:
                            list_type = 'itemize'
                            content_col += f"\\begin{{{list_type}}}\n"
                            in_list = True
                        # Convert bullet to \item
                        bullet_content = re.sub(r'^[-•]\s*', '', line.strip())
                        content_col += f"\\item {bullet_content}\n"
                        i += 1
                        continue

                    # Handle plain text lines
                    if in_list:
                        # Close list before plain text
                        content_col += f"\\end{{{list_type}}}\n\n"
                        in_list = False
                        list_type = None

                    # Skip file directives in content
                    if '\\file' in line:
                        i += 1
                        continue

                    is_section_header = line.strip().startswith('\\textbf{')
                    if is_section_header:
                        content_col += f"\n{line}\n\n"
                    else:
                        content_col += f"{line}\n"
                    i += 1

                # Close any open environments
                if in_list:
                    content_col += f"\\end{{{list_type}}}\n\n"
                if in_block:
                    content_col += "\\end{block}\n\n"

                if has_valid_image:
                    new_document_body = content_col
                    new_document_body += "\\end{columns}\n\n"
                else:
                    new_document_body = content_col

                new_document_body += "\\end{frame}\n\n"

                # Add visible notes
                if visible_notes:
                    real_notes = []
                    for note in visible_notes:
                        clean_note = re.sub(r'^[-•]\s*', '', note.strip())
                        if clean_note and clean_note.lower() not in ["no notes for this slide", "no notes", "none"]:
                            real_notes.append(clean_note)

                    if real_notes:
                        new_document_body += "\\note{\n"
                        new_document_body += "\\begin{itemize}\n"
                        for note in real_notes:
                            new_document_body += f"    \\item {note}\n"
                        new_document_body += "\\end{itemize}\n"
                        new_document_body += "}\n\n"

            new_document_body += "\\end{document}\n"

            full_tex = preamble + new_document_body

            self.write(f"\n📊 TeX Generation Statistics:\n", "cyan")
            self.write(f"  • Title page included\n", "green")
            self.write(f"  • Visible slides included: {total_visible_slides}\n", "green")
            if total_skipped_slides > 0:
                self.write(f"  • Masked slides skipped: {total_skipped_slides}\n", "yellow")

            return full_tex

        except Exception as e:
            self.write(f"Error generating TeX: {str(e)}\n", "red")
            import traceback
            traceback.print_exc()
            return None

    def convert_media_to_latex_clean(self, media: str) -> str:
        """Convert media directive to clean LaTeX"""
        if not media or media == "\\None":
            return ""

        # Handle \file directive
        if media.startswith('\\file'):
            file_path = media.replace('\\file', '').strip()
            # Remove any % that might be at the beginning
            file_path = file_path.lstrip('%').strip()
            # Check if it's an image or video
            if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                return f"\\movie[externalviewer]{{\\includegraphics[width=0.8\\textwidth,keepaspectratio]{{{file_path}}}}}{{{file_path}}}"
            else:
                return f"\\includegraphics[width=0.8\\textwidth,keepaspectratio]{{{file_path}}}"

        # Handle \play directive
        elif media.startswith('\\play'):
            play_content = media.replace('\\play', '').strip()
            if play_content.startswith('\\file'):
                file_path = play_content.replace('\\file', '').strip()
                file_path = file_path.lstrip('%').strip()
                return f"\\movie[externalviewer]{{\\includegraphics[width=0.8\\textwidth,keepaspectratio]{{{file_path}}}}}{{{file_path}}}"
            elif play_content.startswith('\\url'):
                url = play_content.replace('\\url', '').strip()
                url = url.lstrip('%').strip()
                return f"\\href{{{url}}}{{\\textcolor{{blue}}{{\\underline{{Click to play video}}}}}}"
            else:
                return f"\\href{{{play_content}}}{{\\textcolor{{blue}}{{\\underline{{Play video}}}}}}"

        # Handle direct URL
        elif media.startswith('http'):
            return f"\\href{{{media}}}{{\\textcolor{{blue}}{{\\underline{{{media}}}}}}}"

        # Handle TikZ marker
        elif media == "\\TikZ":
            return ""

        # Return as-is for other cases
        return media

    def convert_content_line_to_latex(self, line: str) -> str:
        """Convert a content line to proper LaTeX format while preserving structure"""
        line = line.strip()
        if not line:
            return line

        # Handle bullet points (lines starting with - or •)
        if line.startswith('- ') or line.startswith('• '):
            # Remove the bullet prefix and add \item
            content = line[2:].strip()
            return f"\\item {content}"

        # Handle existing LaTeX commands (they should be preserved as-is)
        if line.startswith('\\'):
            return line

        # Handle text with potential LaTeX commands inside
        # Check for common LaTeX patterns that should be preserved
        if any(cmd in line for cmd in ['\\textbf', '\\textit', '\\textcolor', '\\alert', '\\url']):
            return line

        # Plain text - wrap in appropriate environment if needed
        return line

    def get_notes_configuration(self) -> str:
        """Get the notes configuration based on current mode"""
        mode = self.notes_mode.get() if hasattr(self, 'notes_mode') else "both"

        notes_configs = {
            "slides": "\\setbeameroption{hide notes}",
            "notes": "\\setbeameroption{show only notes}",
            "both": "\\setbeameroption{show notes on second screen=right}"
        }

        config = notes_configs.get(mode, notes_configs["both"])

        return f"""
    % Notes configuration
    \\usepackage{{pgfpages}}
    {config}
    \\setbeamertemplate{{note page}}{{\\pagecolor{{yellow!5}}\\insertnote}}
    """

    def write_tex_masking_summary(self, tex_content: str):
        """Write a summary of masked content in the TeX file"""
        import re

        # Count masked lines in the generated TeX
        masked_line_count = len(re.findall(r'^%', tex_content, re.MULTILINE))
        fully_masked_slides = len(re.findall(r'% ========== FULLY MASKED SLIDE ==========', tex_content))

        if fully_masked_slides > 0 or masked_line_count > 0:
            self.write(f"\n📝 Masking preserved in TeX file:\n", "yellow")
            if fully_masked_slides > 0:
                self.write(f"  • {fully_masked_slides} fully masked slides (commented out)\n", "yellow")
            if masked_line_count > 0:
                self.write(f"  • {masked_line_count} masked content lines (commented out)\n", "yellow")
            self.write(f"  • Masked content appears as % comments in the .tex file\n", "cyan")
            self.write(f"  • Unmask to include content in final PDF\n", "cyan")

    def get_custom_preamble(self) -> str:
        """Generate custom preamble with proper styling"""
        try:
            from BeamerSlideGenerator import get_beamer_preamble

            # Ensure we have non-empty values
            title = self.presentation_info.get('title', '').strip()
            subtitle = self.presentation_info.get('subtitle', '').strip()
            author = self.presentation_info.get('author', '').strip()
            institution = self.presentation_info.get('institution', 'Artificial Intelligence Research and Intelligent Systems (airis4D)')
            short_institute = self.presentation_info.get('short_institute', 'airis4D')
            date = self.presentation_info.get('date', '\\today')

            # Use defaults if empty
            if not title:
                title = "Presentation"
            if not author:
                author = "airis4D"

            # Get base preamble
            base_preamble = get_beamer_preamble(
                title, subtitle, author, institution, short_institute, date
            )

            # Process logo if present
            if 'logo' in self.presentation_info and self.presentation_info['logo']:
                preamble = re.sub(r'\\logo{[^}]*}\s*\n?', '', base_preamble)
                doc_pos = preamble.find("\\begin{document}")
                if doc_pos != -1:
                    logo_command = self.presentation_info['logo'] + "\n\n"
                    preamble = preamble[:doc_pos] + logo_command + preamble[doc_pos:]
                else:
                    preamble = base_preamble + "\n" + self.presentation_info['logo'] + "\n"
            else:
                preamble = base_preamble

            return preamble

        except Exception as e:
            print(f"Error generating custom preamble: {e}")
            # Return fallback preamble
            return """\\documentclass[aspectratio=169]{beamer}
    \\usepackage{graphicx}
    \\usepackage{xcolor}
    \\usetheme{Madrid}
    \\title{Presentation}
    \\author{airis4D}
    \\begin{document}
    """

    def generate_pdf(self) -> None:
        """Generate PDF with improved terminal handling, progress feedback, and color integration"""
        if not self.current_file:
            messagebox.showwarning("Warning", "Please save your file first!")
            return

        try:
            # Clear terminal
            self.clear_terminal()

            # Save current state
            self.save_current_slide()

            # Get base filename without extension
            base_filename = os.path.splitext(self.current_file)[0]
            tex_file = base_filename + '.tex'

            # Step 1: Convert text to TeX with color information
            self.write("Step 1: Converting text to TeX with color support...\n", "white")
            self.convert_to_tex()  # This will handle notes mode correctly

            # Step 2: Read the generated TeX file and add color information
            self.write("\nStep 2: Adding color definitions and XOR text color rules...\n", "white")

            if os.path.exists(tex_file):
                with open(tex_file, 'r', encoding='utf-8') as f:
                    tex_content = f.read()

                # Add color information to the TeX file
                enhanced_content = self.add_color_info_to_output(tex_content)

                # Write back with color information
                with open(tex_file, 'w', encoding='utf-8') as f:
                    f.write(enhanced_content)

                self.write("✓ Added TikZ color definitions and XOR text color rules\n", "green")

                # Show what was added
                color_rules = """
    % XOR Text Color Rule for TikZ:
    % - Text color automatically chosen based on background luminance
    % - Use text=white on dark backgrounds (luminance < 0.179)
    % - Use text=black on light backgrounds (luminance > 0.179)
    %
    % Predefined colors with optimal text colors:
    % - airis4d_blue (dark) → text=white
    % - airis4d_green (light) → text=black
    % - airis4d_orange (light) → text=black
    % - airis4d_red (dark) → text=white
    % - airis4d_purple (dark) → text=white
    % - airis4d_teal (dark) → text=white
    % - airis4d_gray (light) → text=black
    """
                self.write(color_rules, "cyan")
            else:
                self.write("⚠ Warning: TeX file not found after conversion\n", "yellow")

            # Step 3: First pdflatex pass
            self.write("\nStep 3: First pdflatex pass...\n", "white")
            success = self.run_pdflatex(tex_file)

            if success:
                # Step 4: Second pdflatex pass for references and color rendering
                self.write("\nStep 4: Second pdflatex pass (resolving references and colors)...\n", "white")
                success = self.run_pdflatex(tex_file)

                if success:
                    # Optional: Third pass for perfect rendering
                    self.write("\nStep 5: Final pdflatex pass (ensuring color consistency)...\n", "white")
                    success = self.run_pdflatex(tex_file)

                    if success:
                        pdf_file = base_filename + '.pdf'
                        if os.path.exists(pdf_file):
                            # Calculate file size
                            size = os.path.getsize(pdf_file)
                            size_str = self.format_file_size(size)

                            self.write("\n" + "="*60 + "\n", "white")
                            self.write("✓ PDF GENERATED SUCCESSFULLY!\n", "green")
                            self.write("="*60 + "\n", "white")
                            self.write(f"📄 PDF File: {os.path.basename(pdf_file)}\n", "white")
                            self.write(f"📏 Size: {size_str}\n", "white")
                            self.write(f"📍 Location: {os.path.dirname(pdf_file)}\n", "white")

                            # Show color features included
                            self.write("\n🎨 Color Features Included:\n", "cyan")
                            self.write("  • XOR-based text color selection\n", "cyan")
                            self.write("  • Predefined airis4D color palette\n", "cyan")
                            self.write("  • Automatic black/white text contrast\n", "cyan")
                            self.write("  • TikZ color definitions\n", "cyan")

                            # Check for any warnings in the log file
                            log_file = base_filename + '.log'
                            if os.path.exists(log_file):
                                warnings = self.check_latex_log(log_file)
                                if warnings:
                                    self.write(f"\n⚠ Found {warnings} LaTeX warning(s) - check log file\n", "yellow")

                            # Generate color usage report
                            self.generate_color_report(tex_file)

                            # Ask if user wants to view the PDF
                            if messagebox.askyesno("PDF Generation Complete",
                                                 f"✅ PDF generated successfully!\n\n"
                                                 f"File: {os.path.basename(pdf_file)}\n"
                                                 f"Size: {size_str}\n\n"
                                                 f"Color features included:\n"
                                                 f"• XOR text color rules\n"
                                                 f"• Predefined color palette\n"
                                                 f"• Automatic contrast adjustment\n\n"
                                                 f"Would you like to view the PDF now?"):
                                self.preview_pdf(pdf_file)

                            # Offer to open the TeX file to see color definitions
                            if messagebox.askyesno("View Color Definitions",
                                                 "Would you like to see the color definitions added to your TeX file?"):
                                self.view_color_definitions(tex_file)
                        else:
                            self.write("\n✗ Error: PDF file not found after compilation\n", "red")
                    else:
                        self.write("\n✗ Error in final pdflatex pass\n", "red")
                else:
                    self.write("\n✗ Error in second pdflatex pass\n", "red")
            else:
                self.write("\n✗ Error in first pdflatex pass\n", "red")

        except Exception as e:
            error_msg = f"\n✗ Error generating PDF: {str(e)}\n"
            self.write(error_msg, "red")

            # Add detailed error information
            if hasattr(e, '__traceback__'):
                self.write("\nDetailed error information:\n", "red")
                self.write(traceback.format_exc(), "red")

            messagebox.showerror("Error", f"Error generating PDF:\n{str(e)}")

    def add_color_info_to_output(self, tex_content):
        """Add color information and XOR rules to TeX output"""

        # Color information to add
        color_info = """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% COLOR DEFINITIONS FOR TIKZ FIGURES
    %% XOR Rule: Automatic text color selection based on background
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    \\usepackage{xcolor}  % Required for color support

    %% Predefined color palette with optimal text colors
    \\definecolor{airis4d_blue}{RGB}{41,128,185}    % Dark → Use text=white
    \\definecolor{airis4d_green}{RGB}{39,174,96}    % Light → Use text=black
    \\definecolor{airis4d_orange}{RGB}{243,156,18}  % Light → Use text=black
    \\definecolor{airis4d_red}{RGB}{231,76,60}      % Dark → Use text=white
    \\definecolor{airis4d_purple}{RGB}{155,89,182}  % Dark → Use text=white
    \\definecolor{airis4d_teal}{RGB}{26,188,156}    % Dark → Use text=white
    \\definecolor{airis4d_gray}{RGB}{149,165,166}   % Light → Use text=black

    %% Additional utility colors
    \\definecolor{airis4d_lightblue}{RGB}{173,216,230}  % Light → text=black
    \\definecolor{airis4d_darkblue}{RGB}{0,0,139}       % Dark → text=white
    \\definecolor{airis4d_lightgreen}{RGB}{144,238,144} % Light → text=black
    \\definecolor{airis4d_darkgreen}{RGB}{0,100,0}      % Dark → text=white

    %% XOR TEXT COLOR RULE
    %%
    %% For TikZ figures with colored backgrounds:
    %% 1. Calculate background luminance: L = 0.2126*R + 0.7152*G + 0.0722*B
    %% 2. If L > 0.179: Use text=black (light backgrounds)
    %% 3. If L <= 0.179: Use text=white (dark backgrounds)
    %%
    %% Example usage in TikZ:
    %% \\node[fill=airis4d_blue, text=white, minimum width=3cm] {White text on blue};
    %% \\node[fill=airis4d_orange, text=black, minimum width=3cm] {Black text on orange};
    %%
    %% For custom colors, use the get_xor_text_color() function to determine
    %% the optimal text color for any background.

    %% Color helper macros
    \\newcommand{\\airisblue}[1]{\\textcolor{airis4d_blue}{#1}}
    \\newcommand{\\airisgreen}[1]{\\textcolor{airis4d_green}{#1}}
    \\newcommand{\\airisorange}[1]{\\textcolor{airis4d_orange}{#1}}
    \\newcommand{\\airisred}[1]{\\textcolor{airis4d_red}{#1}}
    \\newcommand{\\airispurple}[1]{\\textcolor{airis4d_purple}{#1}}

    %% Color box environments
    \\newenvironment{airisbluebox}{%
        \\begin{tcolorbox}[colback=airis4d_blue!10,colframe=airis4d_blue,title={\\airisblue{Information}}]
    }{\\end{tcolorbox}}

    \\newenvironment{airisgreenbox}{%
        \\begin{tcolorbox}[colback=airis4d_green!10,colframe=airis4d_green,title={\\airisgreen{Note}}]
    }{\\end{tcolorbox}}

    \\newenvironment{airisredbox}{%
        \\begin{tcolorbox}[colback=airis4d_red!10,colframe=airis4d_red,title={\\airisred{Warning}}]
    }{\\end{tcolorbox}}

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

        # Insert color info after documentclass but before begin{document}
        if "\\documentclass" in tex_content:
            # Find the position after documentclass and any packages
            docclass_end = tex_content.find("\\begin{document}")

            if docclass_end != -1:
                # Insert color info right before begin{document}
                new_content = tex_content[:docclass_end] + color_info + tex_content[docclass_end:]
                return new_content

        # Fallback: prepend if we can't find the right position
        return color_info + tex_content

    def generate_color_report(self, tex_file):
        """Generate a report of color usage in the document"""
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Count color occurrences
            colors = {
                'airis4d_blue': content.count('airis4d_blue'),
                'airis4d_green': content.count('airis4d_green'),
                'airis4d_orange': content.count('airis4d_orange'),
                'airis4d_red': content.count('airis4d_red'),
                'airis4d_purple': content.count('airis4d_purple'),
                'airis4d_teal': content.count('airis4d_teal'),
                'airis4d_gray': content.count('airis4d_gray'),
            }

            total_colors = sum(colors.values())

            if total_colors > 0:
                self.write("\n📊 Color Usage Report:\n", "cyan")
                for color, count in colors.items():
                    if count > 0:
                        # Determine recommended text color
                        text_color = get_xor_text_color(color)
                        self.write(f"  • {color}: {count} uses → text={text_color}\n", "cyan")

                self.write(f"  Total color references: {total_colors}\n", "cyan")

        except Exception as e:
            self.write(f"⚠ Could not generate color report: {str(e)}\n", "yellow")

    def view_color_definitions(self, tex_file):
        """Open a window to view color definitions"""
        try:
            with open(tex_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract color definitions section
            import re
            color_section_match = re.search(r'%%%% COLOR DEFINITIONS FOR TIKZ FIGURES(.*?)(?=%%%%|$)',
                                           content, re.DOTALL)

            if color_section_match:
                color_text = color_section_match.group(1).strip()

                # Create a dialog to show color definitions
                dialog = ctk.CTkToplevel(self)
                dialog.title("Color Definitions in TeX File")
                dialog.geometry("700x500")

                # Center dialog
                dialog.transient(self)

                # Create text widget with syntax highlighting
                text_widget = ctk.CTkTextbox(dialog, font=("Courier", 10))
                text_widget.pack(fill="both", expand=True, padx=10, pady=10)

                # Insert color definitions
                text_widget.insert('1.0', color_text)

                # Add copy button
                button_frame = ctk.CTkFrame(dialog)
                button_frame.pack(fill="x", padx=10, pady=10)

                ctk.CTkButton(
                    button_frame,
                    text="Copy to Clipboard",
                    command=lambda: self.copy_text_to_clipboard(color_text)
                ).pack(side="left", padx=5)

                ctk.CTkButton(
                    button_frame,
                    text="Close",
                    command=dialog.destroy
                ).pack(side="right", padx=5)

        except Exception as e:
            messagebox.showerror("Error", f"Could not view color definitions: {str(e)}")

    def copy_text_to_clipboard(self, text):
        """Copy text to clipboard"""
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copied", "Color definitions copied to clipboard")

    def check_latex_log(self, log_file: str) -> None:
        """Check LaTeX log file for warnings and errors"""
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()

            # Check for common issues
            warnings = []
            errors = []

            warning_patterns = [
                r'Warning: ([^\n]+)',
                r'LaTeX Warning: ([^\n]+)',
                r'Package [^\n]+ Warning: ([^\n]+)'
            ]

            error_patterns = [
                r'! ([^\n]+)',
                r'Error: ([^\n]+)',
                r'! LaTeX Error: ([^\n]+)'
            ]

            for pattern in warning_patterns:
                for match in re.finditer(pattern, log_content):
                    warnings.append(match.group(1))

            for pattern in error_patterns:
                for match in re.finditer(pattern, log_content):
                    errors.append(match.group(1))

            if warnings or errors:
                self.write("\nCompilation Report:\n", "yellow")

                if errors:
                    self.write("\nErrors:\n", "red")
                    for error in errors:
                        self.write(f"• {error}\n", "red")

                if warnings:
                    self.write("\nWarnings:\n", "yellow")
                    for warning in warnings:
                        self.write(f"• {warning}\n", "yellow")

        except Exception as e:
            self.write(f"\nError reading log file: {str(e)}\n", "red")

    def write(self, text: str, color: str = "white") -> None:
        """Write text to terminal with color support"""
        try:
            if hasattr(self, 'terminal'):
                self.terminal.write(text, color)
        except Exception as e:
            print(f"Error writing to terminal: {str(e)}", file=sys.__stdout__)


    def run_pdflatex(self, tex_file: str, timeout: int = 60) -> dict:
        """
        Run pdflatex with interactive error editor.
        """
        result = {
            'success': False,
            'errors': [],
            'warnings': [],
            'pdf_path': None,
            'fatal_error': False
        }

        original_dir = os.getcwd()
        max_attempts = 5
        attempt = 0
        last_error_key = None

        try:
            tex_dir = os.path.dirname(tex_file) or '.'
            os.chdir(tex_dir)

            while attempt < max_attempts:
                attempt += 1

                self.write("\n" + "="*60 + "\n", "cyan")
                self.write(f"RUNNING PDFLATEX (Attempt {attempt})\n", "cyan")
                self.write("="*60 + "\n\n", "cyan")

                cmd = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex_file]

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    errors='replace'
                )

                output_lines = []
                error_found = False
                error_line_num = None
                error_message = ""
                error_context = []
                in_error = False

                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break

                    if line:
                        output_lines.append(line.rstrip())

                        if line.startswith('!'):
                            error_found = True
                            error_message = line.strip()
                            error_context = [line.strip()]
                            in_error = True
                            self.write(f"\n⚠ Error detected: {error_message[:100]}\n", "red")
                            continue

                        if in_error:
                            error_context.append(line.rstrip())

                            if re.search(r'l\.\d+', line):
                                match = re.search(r'l\.(\d+)', line)
                                if match:
                                    error_line_num = int(match.group(1))

                            if re.search(r'line\s+\d+', line.lower()):
                                match = re.search(r'line\s+(\d+)', line.lower())
                                if match:
                                    error_line_num = int(match.group(1))

                            if len(error_context) >= 8 or (line.strip() == '' and len(error_context) > 2):
                                in_error = False

                        elif 'Warning' in line:
                            self.write(line, "yellow")
                            result['warnings'].append(line)
                        elif not any(ignore in line for ignore in ['Overfull', 'Underfull', 'pdfTeX warning']):
                            self.write(line, "white")

                        # Use a safe way to update UI
                        try:
                            if hasattr(self, 'root'):
                                self.root.update_idletasks()
                            elif hasattr(self, 'update_idletasks'):
                                self.update_idletasks()
                        except:
                            pass

                if error_found and error_line_num is None:
                    log_file = tex_file.replace('.tex', '.log')
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                log_content = f.read()
                            match = re.search(r'l\.(\d+)', log_content)
                            if match:
                                error_line_num = int(match.group(1))
                        except:
                            pass

                if not error_found:
                    pdf_file = tex_file.replace('.tex', '.pdf')
                    if os.path.exists(pdf_file) and os.path.getsize(pdf_file) > 0:
                        result['success'] = True
                        result['pdf_path'] = pdf_file
                        size = os.path.getsize(pdf_file)
                        self.write(f"\n✓ PDF generated: {os.path.basename(pdf_file)} ({self.format_file_size(size)})\n", "green")
                        return result
                    else:
                        self.write("\n✗ No PDF file generated\n", "red")
                        return result

                error_key = f"{error_message[:50]}_{error_line_num}"

                if last_error_key == error_key:
                    self.write(f"\n⚠ Same error persists after fix attempt.\n", "red")
                    self.write(f"  Error: {error_message[:200]}\n", "yellow")
                    if error_line_num:
                        self.write(f"  Line: {error_line_num}\n", "yellow")

                    response = messagebox.askyesno("Error Persists",
                                                 f"The same error persists.\n\n"
                                                 f"Error: {error_message[:200]}\n"
                                                 f"Line: {error_line_num if error_line_num else 'Unknown'}\n\n"
                                                 f"Do you want to open the editor again?")
                    if not response:
                        result['fatal_error'] = True
                        return result

                last_error_key = error_key

                self.write("\n" + "="*60 + "\n", "yellow")
                self.write("LAUNCHING ERROR EDITOR\n", "yellow")
                self.write("="*60 + "\n", "yellow")
                self.write("The error editor will open. Fix the issue, save, and close.\n", "white")
                self.write("Options: Fix & Recompile | Skip Error | Abort\n\n", "white")

                # Get the parent window safely - try multiple approaches
                parent_window = None
                if hasattr(self, 'root') and self.root:
                    parent_window = self.root
                elif hasattr(self, 'master') and self.master:
                    parent_window = self.master
                elif hasattr(self, 'winfo_toplevel'):
                    parent_window = self
                else:
                    # Create a temporary root if needed
                    parent_window = tk.Tk()
                    parent_window.withdraw()

                # Create and show editor dialog
                editor = LaTeXErrorEditor(
                    parent_window,
                    tex_file,
                    error_line_num or 1,
                    error_message,
                    '\n'.join(error_context),
                    '\n'.join(output_lines[-50:])
                )

                # Wait for editor to close
                try:
                    if hasattr(parent_window, 'wait_window'):
                        parent_window.wait_window(editor)
                    else:
                        editor.wait_window()
                except:
                    pass

                if editor.result == 'fixed':
                    self.write("\n✓ User fixed the file. Recompiling...\n", "green")
                    continue

                elif editor.result == 'skip':
                    self.write("\n⚠ Skipping error...\n", "yellow")
                    if error_line_num:
                        success = self.comment_out_line_in_file(tex_file, error_line_num)
                        if success:
                            self.write(f"✓ Auto-commented line {error_line_num}\n", "green")
                            continue
                    self.write("✗ Could not auto-comment - aborting\n", "red")
                    result['fatal_error'] = True
                    return result

                elif editor.result == 'abort':
                    self.write("\n✗ Compilation aborted by user\n", "red")
                    result['fatal_error'] = True
                    return result

            if attempt >= max_attempts:
                self.write(f"\n✗ Too many attempts ({max_attempts}). Compilation aborted.\n", "red")
                result['fatal_error'] = True

            return result

        except Exception as e:
            self.write(f"\nProcess error: {str(e)}\n", "red")
            result['errors'].append(str(e))
            result['fatal_error'] = True
            return result

        finally:
            try:
                os.chdir(original_dir)
            except:
                pass


    def comment_out_line_in_file(self, tex_file_path, line_num):
        """
        Automatically comment out a specific line in the tex file.
        Returns True if successful, False otherwise.
        """
        try:
            with open(tex_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            if 1 <= line_num <= len(lines):
                original_line = lines[line_num - 1]
                # Check if already commented
                if not original_line.strip().startswith('%'):
                    lines[line_num - 1] = f'% {original_line.rstrip()}  % Auto-commented by error handler\n'

                    with open(tex_file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    self.write(f"  Commented: {original_line[:80].strip()}\n", "yellow")
                    return True
                else:
                    self.write(f"  Line {line_num} already commented\n", "yellow")
                    return True
            else:
                self.write(f"  Invalid line number: {line_num}\n", "red")
                return False

        except Exception as e:
            self.write(f"  Error commenting line: {e}\n", "red")
            return False


    def handle_pdf_completion(self, pdf_file: str, size_str: str) -> None:
        """Handle successful PDF generation"""
        if messagebox.askyesno("Success",
                              f"PDF generated successfully!\nSize: {size_str}\n\nWould you like to view it now?"):
            self.preview_pdf()

    def handle_compilation_error(self, error: Exception) -> None:
        """Handle compilation errors"""
        error_msg = f"Error generating PDF: {str(error)}\n"
        self.write_to_terminal(f"\n✗ {error_msg}", "red")

        # Add detailed error information to terminal
        if hasattr(error, '__traceback__'):
            self.write_to_terminal("\nDetailed error information:\n", "red")
            self.write_to_terminal(traceback.format_exc(), "red")

        messagebox.showerror("Error", f"Error generating PDF:\n{str(error)}")
#------------------------------------------------------------------------------------------------------------------
    def compile_presentation(self, mode: str = "both") -> None:
        """
        Compile presentation with specified notes mode.
        """
        if not self.current_file:
            messagebox.showwarning("Warning", "Please save your file first!")
            return

        try:
            # Save current state
            self.save_current_slide()
            self.save_file()

            # Get base filename
            base_filename = os.path.splitext(self.current_file)[0]
            tex_file = base_filename + '.tex'

            # Clear terminal
            self.clear_terminal()

            # Convert to TEX if needed
            if not os.path.exists(tex_file):
                self.write("Converting text to TEX...\n", "white")
                self.convert_to_tex()

            # Compile with specified mode
            self.write(f"\nCompiling presentation in {mode} mode...\n", "white")
            try:
                pdf_file = compile_with_notes_mode(tex_file, mode)

                if os.path.exists(pdf_file):
                    size = os.path.getsize(pdf_file)
                    size_str = self.format_file_size(size)

                    self.write(f"\n✓ PDF generated successfully ({size_str})\n", "green")

                    if messagebox.askyesno("Success",
                        f"PDF generated successfully in {mode} mode!\n"
                        f"Size: {size_str}\n\n"
                        "Would you like to view it now?"):
                        self.preview_pdf(pdf_file)
                else:
                    self.write("\n✗ Error: PDF file not found after compilation\n", "red")

            except Exception as e:
                self.write(f"\n✗ Error in compilation: {str(e)}\n", "red")
                messagebox.showerror("Error", f"Error compiling PDF:\n{str(e)}")

        except Exception as e:
            self.write(f"\n✗ Error: {str(e)}\n", "red")
            messagebox.showerror("Error", f"Error generating PDF:\n{str(e)}")

    def set_notes_mode(self, mode: str) -> None:
        """Set notes mode and update UI"""
        self.notes_mode.set(mode)
        self.update_notes_buttons(mode)

        # Update preview if available
        if self.current_file:
            self.compile_presentation(mode)

    def preview_pdf(self, pdf_file: str = None) -> None:
        """Preview generated PDF using system default PDF viewer"""
        if not pdf_file:
            if not self.current_file:
                messagebox.showwarning("Warning", "Please save and generate PDF first!")
                return

            # Use default PDF name
            pdf_file = os.path.splitext(self.current_file)[0] + '.pdf'

        if os.path.exists(pdf_file):
            if sys.platform.startswith('win'):
                os.startfile(pdf_file)
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', pdf_file])
            else:
                subprocess.run(['xdg-open', pdf_file])
        else:
            messagebox.showwarning("Warning", "PDF file not found. Generate it first!")

#------------------------------------------------------------------------------------------------------------------

    def new_slide(self) -> None:
        """Create new slide, preserving content if editing first slide"""
        # Check if we're editing the initial slide without saving
        if (self.current_slide_index == 0 and
            self.slides[0]['title'] == 'Presentation Title' and
            len(self.slides[0]['content']) == 1 and
            self.slides[0]['content'][0] == '- First bullet point'):

            # This is the initial template slide, save current content first
            self.save_current_slide()

        # Proceed with creating new slide
        new_slide = {
            'title': 'New Slide',
            'media': '',
            'content': [],
            'notes': []
        }

        self.slides.append(new_slide)
        self.current_slide_index = len(self.slides) - 1
        self.update_slide_list()
        self.load_slide(self.current_slide_index)

        # Focus title entry for immediate editing
        self.title_entry.focus_set()
        self.title_entry.select_range(0, 'end')


    def move_slide(self, direction: int) -> None:
        """Move current slide up or down"""
        if not self.slides or self.current_slide_index < 0:
            return

        new_index = self.current_slide_index + direction
        if 0 <= new_index < len(self.slides):
            self.save_current_slide()
            self.slides[self.current_slide_index], self.slides[new_index] = \
                self.slides[new_index], self.slides[self.current_slide_index]
            self.current_slide_index = new_index
            self.update_slide_list()
            self.load_slide(self.current_slide_index)



    def on_slide_select(self, event) -> None:
        """Handle slide selection from list"""
        index = self.slide_list.index("@%d,%d" % (event.x, event.y)).split('.')[0]
        index = int(index)
        if 0 <= index < len(self.slides):
            self.save_current_slide()
            self.current_slide_index = index
            self.load_slide(index)
            self.update_slide_list()

    def save_current_slide(self):
        """Save current slide data while preserving hidden status"""
        if not hasattr(self, 'slides') or not self.slides:
            self.slides = []
            self.current_slide_index = -1
            return

        if self.current_slide_index < 0:
            title = self.title_entry.get().strip()
            media = self.media_entry.get().strip()

            # Get content with hidden status tracking
            content_with_hidden = []
            hidden_content_indices = []
            content_text = self.content_editor.get('1.0', 'end-1c')
            content_lines = content_text.split('\n')

            for i, line in enumerate(content_lines):
                line = line.rstrip()
                if not line:
                    continue

                # Check if line is hidden (starts with %)
                # Use strip() to check for % at beginning after whitespace
                stripped = line.lstrip()
                is_hidden = stripped.startswith('%')

                # Remove the % prefix and any following spaces
                if is_hidden:
                    # Remove the first % and any spaces after it
                    clean_line = re.sub(r'^\s*%\s*', '', line)
                else:
                    clean_line = line

                if clean_line:  # Only add non-empty lines
                    content_with_hidden.append(clean_line)
                    if is_hidden:
                        hidden_content_indices.append(len(content_with_hidden) - 1)

            # Get notes with hidden status tracking
            notes_with_hidden = []
            hidden_note_indices = []
            notes_text = self.notes_editor.get('1.0', 'end-1c')
            notes_lines = notes_text.split('\n')

            for i, line in enumerate(notes_lines):
                line = line.rstrip()
                if not line:
                    continue

                stripped = line.lstrip()
                is_hidden = stripped.startswith('%')

                if is_hidden:
                    clean_line = re.sub(r'^\s*%\s*', '', line)
                else:
                    clean_line = line

                if clean_line:
                    notes_with_hidden.append(clean_line)
                    if is_hidden:
                        hidden_note_indices.append(len(notes_with_hidden) - 1)

            # Check if media is masked
            media_masked = media.startswith('%') if media else False
            if media_masked:
                media = media.lstrip('%').lstrip()

            if title or media or content_with_hidden or notes_with_hidden:
                new_slide = {
                    'title': title or 'New Slide',
                    'media': media,
                    'content': content_with_hidden,
                    'notes': notes_with_hidden,
                    '_hidden_content_indices': hidden_content_indices,
                    '_hidden_note_indices': hidden_note_indices,
                    '_media_masked': media_masked,
                    '_fully_masked': False
                }
                self.slides.append(new_slide)
                self.current_slide_index = len(self.slides) - 1
                self.update_slide_list()
            return

        # Normal slide save for existing slides
        title = self.title_entry.get().strip()
        media = self.media_entry.get().strip()

        # Check if media is masked
        media_masked = media.startswith('%') if media else False
        if media_masked:
            media = media.lstrip('%').lstrip()

        # Get content with hidden status tracking
        content_with_hidden = []
        hidden_content_indices = []
        content_text = self.content_editor.get('1.0', 'end-1c')
        content_lines = content_text.split('\n')

        for i, line in enumerate(content_lines):
            line = line.rstrip()
            if not line:
                continue

            stripped = line.lstrip()
            is_hidden = stripped.startswith('%')

            if is_hidden:
                clean_line = re.sub(r'^\s*%\s*', '', line)
            else:
                clean_line = line

            if clean_line:
                content_with_hidden.append(clean_line)
                if is_hidden:
                    hidden_content_indices.append(len(content_with_hidden) - 1)

        # Get notes with hidden status tracking
        notes_with_hidden = []
        hidden_note_indices = []
        notes_text = self.notes_editor.get('1.0', 'end-1c')
        notes_lines = notes_text.split('\n')

        for i, line in enumerate(notes_lines):
            line = line.rstrip()
            if not line:
                continue

            stripped = line.lstrip()
            is_hidden = stripped.startswith('%')

            if is_hidden:
                clean_line = re.sub(r'^\s*%\s*', '', line)
            else:
                clean_line = line

            if clean_line:
                notes_with_hidden.append(clean_line)
                if is_hidden:
                    hidden_note_indices.append(len(notes_with_hidden) - 1)

        # Update the slide - preserve hidden status
        if 0 <= self.current_slide_index < len(self.slides):
            self.slides[self.current_slide_index] = {
                'title': title,
                'media': media,
                'content': content_with_hidden,
                'notes': notes_with_hidden,
                '_hidden_content_indices': hidden_content_indices,
                '_hidden_note_indices': hidden_note_indices,
                '_media_masked': media_masked,
                '_fully_masked': self.slides[self.current_slide_index].get('_fully_masked', False)
            }

    def mask_line_in_editor(self, event=None):
        """Mask/unmask the current line in the focused editor (Ctrl+Delete in editors)"""
        focused_widget = self.focus_get()

        # Determine which editor has focus
        editor = None
        editor_name = ""

        if focused_widget == self.content_editor._textbox:
            editor = self.content_editor._textbox
            editor_name = "content"
        elif focused_widget == self.notes_editor._textbox:
            editor = self.notes_editor._textbox
            editor_name = "notes"
        else:
            # Not in an editor - show status message
            self.write("Click in content or notes editor first to mask lines\n", "yellow")
            return "break"

        try:
            # Get current cursor position
            current_pos = editor.index("insert")
            line_start = editor.index(f"{current_pos} linestart")
            line_end = editor.index(f"{current_pos} lineend")

            # Get the line content
            line_content = editor.get(line_start, line_end)

            # Skip empty lines
            if not line_content.strip():
                self.write("Cannot mask empty line\n", "yellow")
                return "break"

            # Check if line is already masked
            if line_content.lstrip().startswith('%'):
                # Unmask: remove the % prefix
                # Remove all leading % and spaces
                cleaned_line = re.sub(r'^\s*%+\s*', '', line_content)
                editor.delete(line_start, line_end)
                editor.insert(line_start, cleaned_line)
                self.write(f"✓ Unmasked line in {editor_name} editor\n", "green")
            else:
                # Mask: add % at beginning
                # Preserve indentation by adding % after spaces?
                indent_match = re.match(r'^(\s*)', line_content)
                indent = indent_match.group(1) if indent_match else ""
                rest = line_content[len(indent):]
                masked_line = f"{indent}% {rest}"
                editor.delete(line_start, line_end)
                editor.insert(line_start, masked_line)
                self.write(f"✓ Masked line in {editor_name} editor\n", "yellow")

            # Preserve cursor position approximately at same column
            cursor_col = int(editor.index("insert").split('.')[1])
            new_pos = f"{line_start}+{min(cursor_col, len(masked_line))}c"
            editor.mark_set("insert", new_pos)

            # Update syntax highlighting
            if hasattr(self, 'syntax_highlighter') and self.syntax_highlighter.active:
                self.syntax_highlighter.highlight()

            # Mark that changes need saving
            self.save_current_slide()

        except Exception as e:
            self.write(f"Error masking line: {str(e)}\n", "red")
            import traceback
            traceback.print_exc()

        return "break"

    def clear_editor(self) -> None:
        """Clear editor fields"""
        self.title_entry.delete(0, 'end')
        self.media_entry.delete(0, 'end')
        self.content_editor.delete('1.0', 'end')

    # Media Handling
    def browse_media(self) -> None:
        """Browse media files with thumbnail preview"""
        def on_file_selected(media_path):
            self.media_entry.delete(0, 'end')
            self.media_entry.insert(0, media_path)

        browser = FileThumbnailBrowser(self, callback=on_file_selected)
        browser.transient(self)
        browser.grab_set()
        self.wait_window(browser)

    def youtube_dialog(self) -> None:
        """Handle YouTube video insertion"""
        dialog = ctk.CTkInputDialog(
            text="Enter YouTube URL:",
            title="Add YouTube Video"
        )
        url = dialog.get_input()
        if url:
            if 'youtube.com' in url or 'youtu.be' in url:
                self.media_entry.delete(0, 'end')
                self.media_entry.insert(0, f"\\play {url}")
            else:
                messagebox.showwarning("Invalid URL", "Please enter a valid YouTube URL")

    def search_images(self) -> None:
        """Open image search for current slide"""
        query = construct_search_query(
            self.title_entry.get(),
            self.content_editor.get("1.0", "end").split('\n')
        )
        open_google_image_search(query)

    def generate_tex_content(self) -> str:
        """Generate complete tex file content with proper notes handling"""
        # Get base content
        if hasattr(self, 'custom_preamble'):
            content = self.custom_preamble
        else:
            content = get_beamer_preamble(
                self.presentation_info['title'],
                self.presentation_info['subtitle'],
                self.presentation_info['author'],
                self.presentation_info['institution'],
                self.presentation_info['short_institute'],
                self.presentation_info['date']
            )

        # Modify preamble for notes configuration
        content = self.modify_preamble_for_notes(content)
        print(content)
        # Add slides with appropriate notes handling
        for slide in self.slides:
            content += f"\\begin{frame}\n"
            content += f"\\frametitle{{{slide['title']}}}\n"

            if slide['media']:
                content += f"{slide['media']}\n"

            for item in slide['content']:
                if item.strip():
                    content += f"{item}\n"

            content += "\\end{frame}\n"

            # Add notes if not in slides_only mode
            if self.notes_mode.get() != "slides_only" and 'notes' in slide and slide['notes']:
                content += "\\note{\n\\begin{itemize}\n"
                for note in slide['notes']:
                    if note.strip():
                        note = note.lstrip('•- ').strip()
                        content += f"\\item {note}\n"
                content += "\\end{itemize}\n}\n"
            note=""
            content += "\n"

        content += "\\end{document}\n"
        return content

    def modify_preamble_for_notes(self, tex_content: str) -> str:
        """Modify the preamble based on current notes mode"""
        mode = self.notes_mode.get()
        print(mode)
        # Define the notes configuration based on mode
        notes_configs = {
            "slides": "\\setbeameroption{hide notes}",
            "notes": "\\setbeameroption{show only notes}",
            "both": "\\setbeameroption{show notes on second screen=right}"
        }

        # First, remove any existing notes configurations
        tex_content = re.sub(r'%.*\\setbeameroption{[^}]*}.*\n', '', tex_content)
        tex_content = re.sub(r'\\setbeameroption{[^}]*}', '', tex_content)

        # Ensure pgfpages package is present
        if "\\usepackage{pgfpages}" not in tex_content:
            package_line = "\\usepackage{pgfpages}\n"
        else:
            package_line = ""

        # Get the appropriate notes configuration
        notes_config = notes_configs[mode]

        # Add the configuration just before \begin{document}
        doc_pos = tex_content.find("\\begin{document}")
        if doc_pos != -1:
            insert_text = f"{package_line}% Notes configuration\n{notes_config}\n\\setbeamertemplate{{note page}}{{\\pagecolor{{yellow!5}}\\insertnote}}\n\n"
            tex_content = tex_content[:doc_pos] + insert_text + tex_content[doc_pos:]

        return tex_content

    def load_file(self, filename: str) -> None:
        """Load presentation from file with full preservation of masked content"""
        try:
            logger.info(f"Loading file: {filename}")

            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            self.slides = []
            self.current_slide_index = -1

            # Extract presentation info
            import re
            for key in self.presentation_info:
                pattern = f"\\\\{key}{{(.*?)}}"
                match = re.search(pattern, content)
                if match:
                    self.presentation_info[key] = match.group(1)

            # Split into slides by finding all title lines (including commented ones)
            lines = content.split('\n')

            slides_raw = []
            current_slide_lines = []

            for line in lines:
                # Match both "\title" and "% \title" (with optional space after %)
                if re.match(r'^%?\s*\\title\s+', line):
                    if current_slide_lines:
                        slides_raw.append(current_slide_lines)
                    current_slide_lines = [line]
                else:
                    current_slide_lines.append(line)

            if current_slide_lines:
                slides_raw.append(current_slide_lines)

            logger.info(f"Found {len(slides_raw)} slide blocks")

            masked_slides_count = 0
            total_masked_lines = 0

            for slide_idx, slide_lines in enumerate(slides_raw):
                # Check if slide is fully masked
                non_empty = [l for l in slide_lines if l.strip()]
                is_fully_masked = False

                if non_empty:
                    # Check first non-empty line - if it starts with % and contains \title
                    first_line = non_empty[0].strip()
                    if first_line.startswith('%') and '\\title' in first_line:
                        # Check if ALL non-empty lines start with %
                        is_fully_masked = all(l.strip().startswith('%') for l in non_empty)

                # Extract title - handle both commented and uncommented
                title_line = None
                for line in slide_lines:
                    # Match title even if preceded by % and optional spaces
                    title_match = re.search(r'%?\s*\\title\s+(.+)$', line)
                    if title_match:
                        title_line = title_match.group(1).strip()
                        break

                if title_line is None:
                    logger.warning(f"No title found in slide block {slide_idx}")
                    continue

                title = title_line

                if is_fully_masked and not title.startswith('[DELETED]'):
                    title = f"[DELETED] {title}"
                    masked_slides_count += 1

                # Parse content and notes blocks
                in_content = False
                in_notes = False
                content_lines = []
                notes_lines = []
                hidden_content_indices = []
                hidden_note_indices = []
                media = ""
                media_masked = False

                for line in slide_lines:
                    # Check for Content block boundaries (handle possible spaces and %)
                    if re.match(r'^%?\s*\\begin{Content}\s*$', line):
                        in_content = True
                        in_notes = False
                        continue
                    elif re.match(r'^%?\s*\\end{Content}\s*$', line):
                        in_content = False
                        continue

                    # Check for Notes block boundaries
                    if re.match(r'^%?\s*\\begin{Notes}\s*$', line):
                        in_notes = True
                        in_content = False
                        continue
                    elif re.match(r'^%?\s*\\end{Notes}\s*$', line):
                        in_notes = False
                        continue

                    # Process content lines
                    if in_content and line.strip():
                        is_masked = line.lstrip().startswith('%')
                        # Remove leading % and spaces
                        clean_line = re.sub(r'^\s*%\s*', '', line) if is_masked else line

                        if clean_line.strip():
                            # Check if this is a media directive (first line starting with \)
                            if not media and clean_line.strip().startswith('\\'):
                                media = clean_line.strip()
                                media_masked = is_masked
                                if is_masked:
                                    total_masked_lines += 1
                                logger.info(f"  Media found: {media[:50]} (masked={media_masked})")
                            else:
                                content_lines.append(clean_line.rstrip())
                                if is_masked:
                                    hidden_content_indices.append(len(content_lines) - 1)
                                    total_masked_lines += 1
                                    logger.info(f"  Masked content line {len(content_lines)}: {clean_line[:50]}")
                                else:
                                    logger.info(f"  Visible content line {len(content_lines)}: {clean_line[:50]}")

                    # Process notes lines
                    if in_notes and line.strip():
                        is_masked = line.lstrip().startswith('%')
                        clean_line = re.sub(r'^\s*%\s*', '', line) if is_masked else line
                        if clean_line.strip():
                            notes_lines.append(clean_line.rstrip())
                            if is_masked:
                                hidden_note_indices.append(len(notes_lines) - 1)
                                total_masked_lines += 1

                # Create slide with preserved masking
                slide = {
                    'title': title,
                    'media': media,
                    'content': content_lines,
                    'notes': notes_lines,
                    '_hidden_content_indices': hidden_content_indices,
                    '_hidden_note_indices': hidden_note_indices,
                    '_media_masked': media_masked,
                    '_fully_masked': is_fully_masked
                }

                self.slides.append(slide)
                logger.info(f"Slide {len(self.slides)}: '{title[:50]}' (fully_masked={is_fully_masked}, content={len(content_lines)}, hidden={len(hidden_content_indices)})")

            if self.slides:
                self.current_slide_index = 0
                self.load_slide(0)
            else:
                logger.error("No slides were loaded!")
                messagebox.showerror("Error", "No slides could be loaded from the file!")
                return

            self.update_slide_list()

            # Show loading summary
            summary_parts = [f"✓ Loaded {len(self.slides)} slides"]
            if masked_slides_count > 0:
                summary_parts.append(f"{masked_slides_count} fully masked")
            if total_masked_lines > 0:
                summary_parts.append(f"{total_masked_lines} masked lines")

            self.write(" | ".join(summary_parts) + "\n", "green")

            if masked_slides_count > 0 or total_masked_lines > 0:
                self.write("ℹ Masked content is shown with strikethrough\n", "yellow")
                self.write("  • Click on masked line + Ctrl+Delete to unmask\n", "cyan")

        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.write(f"✗ {error_msg}\n", "red")
            messagebox.showerror("Error", f"Error loading file:\n{str(e)}")

    def _apply_masking_to_slide(self, slide_index: int) -> None:
        """Apply masking to a slide (mark as deleted)"""
        if 0 <= slide_index < len(self.slides):
            slide = self.slides[slide_index]

            # Add marker to title
            if not slide['title'].startswith('[DELETED]'):
                slide['title'] = f"[DELETED] {slide['title']}"

            # Note: We don't modify content here because it's already loaded
            # The is_slide_masked method will check for the title marker

            # Update the slide list display
            self.update_slide_list()

    def is_slide_masked(self, slide_index: int) -> bool:
        """Check if a slide is masked (deleted)"""
        if not 0 <= slide_index < len(self.slides):
            return False

        slide = self.slides[slide_index]

        # Check title for deletion marker
        if slide.get('title', '').startswith('[DELETED]'):
            return True

        # Check if all content is commented
        content_items = slide.get('content', [])
        if content_items:
            # If all non-empty content lines start with %, it's masked
            non_empty_items = [item for item in content_items if item.strip()]
            if non_empty_items:
                all_commented = all(item.strip().startswith('%') for item in non_empty_items)
                if all_commented:
                    return True

        return False

    def show_settings_dialog(self) -> None:
        """Show presentation settings dialog with logo handling"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Presentation Settings")
        dialog.geometry("500x450")

        # Make dialog modal but wait until it's visible
        dialog.transient(self)

        # Center the dialog on parent window
        def center_dialog():
            dialog.update_idletasks()  # Make sure dialog is fully created
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            window_width = dialog.winfo_width()
            window_height = dialog.winfo_height()
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2
            dialog.geometry(f"+{x}+{y}")

            # Set grab after dialog is visible
            dialog.after(100, lambda: dialog.grab_set())

        # Container frame for settings
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create entry fields for presentation metadata
        entries = {}
        row = 0
        for key, value in self.presentation_info.items():
            if key != 'logo':  # Handle logo separately
                label = ctk.CTkLabel(main_frame, text=key.title() + ":")
                label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

                entry = ctk.CTkEntry(main_frame, width=300)
                entry.insert(0, value)
                entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
                entries[key] = entry
                row += 1

        # Add logo selection
        logo_label = ctk.CTkLabel(main_frame, text="Logo:")
        logo_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")

        logo_frame = ctk.CTkFrame(main_frame)
        logo_frame.grid(row=row, column=1, sticky="ew", padx=5, pady=5)

        logo_entry = ctk.CTkEntry(logo_frame, width=240)
        logo_entry.pack(side="left", padx=(0, 5), fill="x", expand=True)

        # Insert existing logo path if present
        if 'logo' in self.presentation_info:
            logo_path = self.presentation_info['logo']
            # Extract path from logo command if present
            if '\\includegraphics' in logo_path:
                match = re.search(r'\\includegraphics\[.*?\]{(.*?)}', logo_path)
                if match:
                    logo_entry.insert(0, match.group(1))
            else:
                logo_entry.insert(0, logo_path)

        def browse_logo():
            """Handle logo file selection"""
            filename = filedialog.askopenfilename(
                title="Select Logo Image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.pdf"),
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg *.jpeg"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ],
                parent=dialog  # Set parent for proper modal behavior
            )
            if filename:
                logo_entry.delete(0, 'end')
                logo_entry.insert(0, filename)

        ctk.CTkButton(logo_frame, text="Browse",
                      command=browse_logo).pack(side="left", padx=5)

        def save_settings():
            """Save settings including logo"""
            try:
                # Save regular metadata
                for key, entry in entries.items():
                    self.presentation_info[key] = entry.get()

                # Save logo if specified
                logo_path = logo_entry.get().strip()
                if logo_path:
                    if os.path.exists(logo_path):
                        # Save logo path with LaTeX formatting
                        self.presentation_info['logo'] = f"\\logo{{\\includegraphics[height=1cm]{{{logo_path}}}}}"
                    else:
                        messagebox.showerror("Error",
                                           f"Logo file not found:\n{logo_path}",
                                           parent=dialog)
                        return
                else:
                    # Remove logo if entry is empty
                    self.presentation_info.pop('logo', None)

                dialog.grab_release()
                dialog.destroy()

            except Exception as e:
                messagebox.showerror("Error",
                                   f"Error saving settings:\n{str(e)}",
                                   parent=dialog)

        def on_cancel():
            """Handle dialog cancellation"""
            dialog.grab_release()
            dialog.destroy()

        # Add buttons frame
        buttons_frame = ctk.CTkFrame(main_frame)
        buttons_frame.grid(row=row + 1, column=0, columnspan=2, pady=20)

        # Add Save and Cancel buttons
        ctk.CTkButton(buttons_frame, text="Save Settings",
                      command=save_settings).pack(side="left", padx=10)

        ctk.CTkButton(buttons_frame, text="Cancel",
                      command=on_cancel).pack(side="left", padx=10)

        # Configure grid
        main_frame.columnconfigure(1, weight=1)

        # Handle dialog close button
        dialog.protocol("WM_DELETE_WINDOW", on_cancel)

        # Center and show dialog
        dialog.after(10, center_dialog)

        # Lift dialog to top
        dialog.lift()

    def debug_slide_data(self):
        """Debug function to print slide data structure"""
        print("\n=== DEBUG: Current Slide Data ===")
        for idx, slide in enumerate(self.slides):
            print(f"\nSlide {idx + 1}:")
            print(f"  Title: {slide.get('title')}")
            print(f"  Media: {slide.get('media')}")
            print(f"  Media masked: {slide.get('_media_masked')}")
            print(f"  Content: {slide.get('content')}")
            print(f"  Hidden content indices: {slide.get('_hidden_content_indices')}")
            print(f"  Notes: {slide.get('notes')}")
            print(f"  Hidden note indices: {slide.get('_hidden_note_indices')}")
            print(f"  Fully masked: {slide.get('_fully_masked')}")
        print("=" * 40)

    def edit_preamble(self):
        """Open preamble editor with logo support"""
        # Get current preamble including logo
        current_preamble = self.get_custom_preamble()

        # Open preamble editor
        new_preamble = PreambleEditor.edit_preamble(self, current_preamble)

        if new_preamble is not None:
            # Store the custom preamble
            self.custom_preamble = new_preamble
            messagebox.showinfo("Success", "Preamble updated successfully!")

    def present_with_notes(self) -> None:
        """Present PDF using pympress for dual-screen display with notes"""
        if not self.current_file:
            messagebox.showwarning("Warning", "Please save your file first!")
            return

        try:
            # Get absolute paths
            base_filename = os.path.splitext(self.current_file)[0]
            pdf_file = base_filename + '.pdf'
            abs_pdf_path = os.path.abspath(pdf_file)

            # Check if PDF exists and generate if needed
            if not os.path.exists(abs_pdf_path):
                self.write_to_terminal("PDF not found. Generating...")
                self.generate_pdf()

                if not os.path.exists(abs_pdf_path):
                    messagebox.showerror("Error", "Failed to generate PDF presentation.")
                    return

            # Use system pympress
            self.write_to_terminal("Launching pympress presentation viewer...")

            # Try different possible pympress locations
            pympress_paths = [
                shutil.which('pympress'),
                '/usr/local/bin/pympress',
                '/usr/bin/pympress',
                os.path.expanduser('~/.local/bin/pympress')
            ]

            launched = False
            for path in pympress_paths:
                if path and os.path.exists(path):
                    subprocess.Popen([path, abs_pdf_path])
                    launched = True
                    break

            if not launched:
                # Try using python -m pympress
                try:
                    subprocess.Popen([sys.executable, "-m", "pympress", abs_pdf_path])
                    launched = True
                except:
                    pass

            if launched:
                self.write_to_terminal("✓ Presentation launched successfully\n", "green")
                self.write_to_terminal("\nPympress Controls:\n")
                self.write_to_terminal("- Right Arrow/Space/Page Down: Next slide\n")
                self.write_to_terminal("- Left Arrow/Page Up: Previous slide\n")
                self.write_to_terminal("- Escape: Exit presentation\n")
                self.write_to_terminal("- F11: Toggle fullscreen\n")
                self.write_to_terminal("- N: Toggle notes\n")
                self.write_to_terminal("- P: Pause/unpause timer\n")
            else:
                self.write_to_terminal("✗ pympress not found. Please install it:\n", "red")
                self.write_to_terminal("  pip install pympress\n", "yellow")

        except Exception as e:
            self.write_to_terminal(f"✗ Error launching presentation: {str(e)}\n", "red")
            messagebox.showerror("Error", f"Error launching presentation:\n{str(e)}")

    def add_tikz_color_helper(self):
        """Add TikZ color helper button to toolbar"""
        # Add to existing toolbar
        if hasattr(self, 'toolbar'):
            tikz_button = ctk.CTkButton(
                self.toolbar,
                text="🎨 TikZ Colors",
                command=self.show_tikz_color_helper,
                width=100,
                fg_color="#9b59b6",
                hover_color="#8e44ad"
            )
            tikz_button.pack(side="left", padx=5)
            self.create_tooltip(tikz_button, "TikZ Color Helper with XOR text coloring")

    def show_tikz_color_helper(self):
        """Show TikZ color helper dialog"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("TikZ Color Helper")
        dialog.geometry("600x500")

        # Center dialog
        dialog.transient(self)
        dialog.grab_set()

        # Create notebook for tabs
        notebook = ctk.CTkTabview(dialog)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tab 1: Color Picker
        color_tab = notebook.add("Color Picker")
        self.create_color_picker_tab(color_tab)

        # Tab 2: Color Definitions
        definitions_tab = notebook.add("Color Definitions")
        self.create_definitions_tab(definitions_tab)

        # Tab 3: Examples
        examples_tab = notebook.add("Examples")
        self.create_examples_tab(examples_tab)

    def create_color_picker_tab(self, parent):
        """Create color picker tab"""
        # Initialize color helper
        self.tikz_color_helper = TikZColorHelper()

        # Instructions
        instructions = ctk.CTkLabel(
            parent,
            text="Select a background color. Text color will be automatically chosen using XOR rule.",
            wraplength=550,
            font=("Arial", 11)
        )
        instructions.pack(pady=10)

        # Color input frame
        input_frame = ctk.CTkFrame(parent)
        input_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(input_frame, text="Background Color:").pack(side="left", padx=5)
        self.bg_color_entry = ctk.CTkEntry(input_frame, width=150)
        self.bg_color_entry.pack(side="left", padx=5)
        self.bg_color_entry.insert(0, "airis4d_blue")

        # Preview button
        preview_btn = ctk.CTkButton(
            input_frame,
            text="Preview",
            command=self.preview_tikz_colors,
            width=80
        )
        preview_btn.pack(side="left", padx=5)

        # Pick color button
        pick_btn = ctk.CTkButton(
            input_frame,
            text="🎨 Pick Color",
            command=self.pick_tikz_color,
            width=100,
            fg_color="#3498db"
        )
        pick_btn.pack(side="left", padx=5)

        # Preview area
        self.tikz_preview_frame = ctk.CTkFrame(parent, height=150)
        self.tikz_preview_frame.pack(fill="x", padx=10, pady=10)
        self.tikz_preview_frame.pack_propagate(False)

        self.tikz_preview_label = ctk.CTkLabel(
            self.tikz_preview_frame,
            text="TikZ Color Preview",
            font=("Arial", 14),
            height=140,
            corner_radius=10
        )
        self.tikz_preview_label.pack(pady=5)

        # Generated code area
        code_frame = ctk.CTkFrame(parent)
        code_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(code_frame, text="Generated TikZ Code:").pack(anchor="w", padx=5, pady=5)

        self.tikz_code_text = ctk.CTkTextbox(code_frame, height=100)
        self.tikz_code_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Insert button
        insert_btn = ctk.CTkButton(
            parent,
            text="Insert into Content",
            command=self.insert_tikz_code,
            width=150,
            fg_color="#2ecc71"
        )
        insert_btn.pack(pady=10)

    def pick_tikz_color(self):
        """Pick a color using color picker"""
        result = ColorPickerDialog.pick_color(self)
        if result:
            self.bg_color_entry.delete(0, 'end')
            self.bg_color_entry.insert(0, result['name'])

            # Add to color helper
            self.tikz_color_helper.define_color(result['name'], result['rgb'])

            self.preview_tikz_colors()

    def preview_tikz_colors(self):
        """Preview TikZ colors"""
        bg_color = self.bg_color_entry.get().strip()

        if not bg_color:
            messagebox.showwarning("Warning", "Please enter a background color")
            return

        # Get text color
        text_color = self.tikz_color_helper.get_text_color(bg_color)

        # Update preview
        try:
            # Try to set background color in preview
            if bg_color in self.tikz_color_helper.color_definitions:
                rgb = self.tikz_color_helper.color_definitions[bg_color]
                hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            elif bg_color in self.tikz_color_helper.default_colors:
                rgb = self.tikz_color_helper.default_colors[bg_color]
                hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            else:
                # Assume it's a valid color name
                hex_color = bg_color

            self.tikz_preview_label.configure(
                text=f"Background: {bg_color}\nText Color: {text_color}",
                fg_color=hex_color,
                text_color="white" if text_color == "white" else "black"
            )

            # Generate TikZ code
            tikz_code = f"""
    % TikZ node with automatic text color
    \\definecolor{{{bg_color}}}{{RGB}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}
    \\node[fill={bg_color}, text={text_color}, minimum width=3cm, minimum height=1cm] {{{text_color.upper()} text on {bg_color}}};
    """

            self.tikz_code_text.delete('1.0', 'end')
            self.tikz_code_text.insert('1.0', tikz_code.strip())

        except Exception as e:
            messagebox.showerror("Error", f"Could not preview color: {str(e)}")

    def insert_tikz_code(self):
        """Insert TikZ code into content editor"""
        code = self.tikz_code_text.get('1.0', 'end-1c')
        if code:
            self.content_editor.insert('insert', f"\n{code}\n")

    def create_definitions_tab(self, parent):
        """Create color definitions tab"""
        text = ctk.CTkTextbox(parent)
        text.pack(fill="both", expand=True, padx=10, pady=10)

        definitions = self.tikz_color_helper.generate_tikz_color_definitions()

        text.insert('1.0', "% TikZ Color Definitions (add to preamble)\n")
        text.insert('end', definitions)

        # Add copy button
        copy_btn = ctk.CTkButton(
            parent,
            text="Copy to Clipboard",
            command=lambda: self.copy_to_clipboard(definitions),
            width=150
        )
        copy_btn.pack(pady=10)

    def create_examples_tab(self, parent):
        """Create examples tab"""
        text = ctk.CTkTextbox(parent)
        text.pack(fill="both", expand=True, padx=10, pady=10)

        examples = """
    % Example 1: Basic colored node
    \\node[fill=airis4d_blue, text=white, draw=black, thick, minimum width=2cm] {White Text};

    % Example 2: Multiple nodes with different colors
    \\begin{tikzpicture}
        \\node[fill=airis4d_green, text=black, circle] at (0,0) {A};
        \\node[fill=airis4d_red, text=white, circle] at (2,0) {B};
        \\node[fill=airis4d_orange, text=black, circle] at (4,0) {C};
    \\end{tikzpicture}

    % Example 3: Using the create_colored_node helper
    % (Include the TikZColorHelper class in your document preamble)
    """

        text.insert('1.0', examples)

    def copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copied", "Color definitions copied to clipboard")

    def add_color_info_to_output(self, content):
        """Add color information to generated output"""
        color_info = """
    % Color Information for TikZ Figures
    % Use these color definitions for consistent theming
    % Text colors are automatically chosen using XOR rule

    % Default color palette
    \\definecolor{airis4d_blue}{RGB}{41,128,185}
    \\definecolor{airis4d_green}{RGB}{39,174,96}
    \\definecolor{airis4d_orange}{RGB}{243,156,18}
    \\definecolor{airis4d_red}{RGB}{231,76,60}
    \\definecolor{airis4d_purple}{RGB}{155,89,182}
    \\definecolor{airis4d_teal}{RGB}{26,188,156}
    \\definecolor{airis4d_gray}{RGB}{149,165,166}

    % XOR Rule for text colors:
    % - Use text=white on dark backgrounds (luminance < 0.179)
    % - Use text=black on light backgrounds (luminance > 0.179)

    % Example usage:
    % \\node[fill=airis4d_blue, text=white] {White text on blue};
    % \\node[fill=airis4d_orange, text=black] {Black text on orange};
    """

        # Insert color info after the document class
        if "\\documentclass" in content:
            parts = content.split("\\documentclass", 1)
            new_content = parts[0] + "\\documentclass" + parts[1].replace(
                "\\begin{document}",
                color_info + "\n\\begin{document}",
                1
            )
            return new_content

        return color_info + "\n" + content

#-----------------------------------------------Help Functions --------------------------------------------
def modify_preamble_for_notes_mode(tex_content: str, mode: str) -> str:
    """
    Modify preamble based on notes mode while preserving original content.

    Args:
        tex_content: Original TEX content
        mode: 'slides', 'notes', or 'both'
    Returns:
        Modified TEX content
    """
    # First find document begin position
    doc_pos = tex_content.find("\\begin{document}")
    if doc_pos == -1:
        return tex_content

    # Split content into preamble and document
    preamble = tex_content[:doc_pos]
    document = tex_content[doc_pos:]

    # Define notes configurations
    notes_configs = {
        "slides": "\\setbeameroption{hide notes}",
        "notes": "\\setbeameroption{show only notes}",
        "both": "\\setbeameroption{show notes on second screen=right}"
    }

    # Remove any existing notes configurations
    preamble = re.sub(r'\\setbeameroption{[^}]*}', '', preamble)

    # Ensure pgfpages package
    if "\\usepackage{pgfpages}" not in preamble:
        preamble = preamble.rstrip() + "\n\\usepackage{pgfpages}\n"

    # Add appropriate notes configuration
    notes_config = notes_configs.get(mode, notes_configs['both'])
    preamble = preamble.rstrip() + f"\n\n% Notes configuration\n{notes_config}\n"

    # Add template style
    preamble += "\\setbeamertemplate{note page}{\\pagecolor{yellow!5}\\insertnote}\n"

    return preamble + document

def compile_with_notes_mode(input_file: str, mode: str, keep_temp: bool = False) -> str:
    """
    Compile TEX file with specified notes mode.

    Args:
        input_file: Path to input TEX file
        mode: 'slides', 'notes', or 'both'
        keep_temp: Whether to keep temporary files
    Returns:
        Path to generated PDF
    """
    try:
        # Create temp directory for compilation
        temp_dir = tempfile.mkdtemp()
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        temp_tex = os.path.join(temp_dir, f"{base_name}_{mode}.tex")
        output_pdf = os.path.join(temp_dir, f"{base_name}_{mode}.pdf")

        # Read original content
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Modify content for notes mode
        modified_content = modify_preamble_for_notes_mode(content, mode)

        # Write modified content to temp file
        with open(temp_tex, 'w', encoding='utf-8') as f:
            f.write(modified_content)

        # Copy required media files to temp directory
        media_dir = os.path.join(os.path.dirname(input_file), 'media_files')
        if os.path.exists(media_dir):
            temp_media = os.path.join(temp_dir, 'media_files')
            shutil.copytree(media_dir, temp_media)

        # Compile document
        os.chdir(temp_dir)
        for _ in range(2):  # Two passes for references
            subprocess.run(['pdflatex', '-interaction=nonstopmode', temp_tex],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Move final PDF to original directory
        final_pdf = os.path.join(os.path.dirname(input_file), f"{base_name}_{mode}.pdf")
        shutil.copy2(output_pdf, final_pdf)

        return final_pdf

    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)







#---------------------------------------------------------------------------------


def verify_pymupdf_installation():
    """
    Verify PyMuPDF is installed correctly and usable.
    """
    try:
        import fitz
        # Try to access version info
        version = fitz.version[0]
        print(f"PyMuPDF version: {version}")

        # Try to create a simple test document
        test_doc = fitz.open()  # Creates an empty PDF
        test_doc.new_page()     # Adds a page
        test_doc.close()        # Closes the document

        print("✓ PyMuPDF verification successful")
        return True
    except Exception as e:
        print(f"PyMuPDF verification failed: {str(e)}")
        return False

def get_pymupdf_info():
    """
    Get detailed information about PyMuPDF installation.
    """
    try:
        import fitz
        return {
            'version': fitz.version[0],
            'binding_version': fitz.version[1],
            'build_date': fitz.version[2],
            'lib_path': fitz.__file__
        }
    except Exception as e:
        return f"Error getting PyMuPDF info: {str(e)}"

def import_required_packages():
    """
    Import all required packages with proper error handling and verification.
    """
    try:
        # First verify PyMuPDF
        if not verify_pymupdf_installation():
            raise ImportError("PyMuPDF installation verification failed")

        # Now import everything else
        import tkinter as tk
        from PIL import Image, ImageDraw, ImageTk
        import customtkinter as ctk
        from tkinter import ttk, filedialog, messagebox, simpledialog
        import screeninfo
        import requests
        import cv2
        import yt_dlp
        import fitz

        modules = {
            'tk': tk,
            'Image': Image,
            'ImageDraw': ImageDraw,
            'ImageTk': ImageTk,
            'ctk': ctk,
            'ttk': ttk,
            'fitz': fitz,
            'screeninfo': screeninfo,
            'requests': requests,
            'cv2': cv2,
            'yt_dlp': yt_dlp
        }

        # Print success message with version info
        print(f"\nSuccessfully imported all packages:")
        print(f"PyMuPDF version: {fitz.version[0]}")
        print(f"PIL version: {Image.__version__}")
        print(f"OpenCV version: {cv2.__version__}")

        return modules

    except Exception as e:
        print(f"Error importing required packages: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def convert_beamer_tex_to_simple_text(tex_file_path):
    """
    Convert Beamer .tex file to simple text format for BeamerSlideGenerator
    Returns path to the generated text file
    """
    import re
    import os
    from pathlib import Path

    try:
        with open(tex_file_path, 'r', encoding='utf-8') as f:
            tex_content = f.read()

        # Create output path
        tex_path = Path(tex_file_path)
        output_path = tex_path.parent / f"{tex_path.stem}_converted.txt"

        def extract_complete_tikz_blocks(text):
            """Extract complete TikZ blocks including surrounding LaTeX commands"""
            tikz_blocks = []

            # Pattern to match TikZ with possible surrounding commands like \scalebox, \centering, etc.
            # This captures things like: \centering\begin{tikzpicture}...\end{tikzpicture}
            tikz_pattern = r'((?:\\centering\s*)?(?:\\scalebox\{[^}]*\}\s*\{)?(?:\\begin\{center\})?(?:\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})(?:\\end\{center\})?(?:\})?)'
            tikz_matches = list(re.finditer(tikz_pattern, text, re.DOTALL))

            # Replace from last to first to avoid index issues
            for i, match in enumerate(reversed(tikz_matches)):
                tikz_code = match.group(1).strip()
                # Store the full TikZ block
                tikz_blocks.append(tikz_code)
                # Replace with unique placeholder
                placeholder = f"{{TIKZ_BLOCK_{len(tikz_matches)-i-1}}}"
                text = text[:match.start()] + placeholder + text[match.end():]

            return text, tikz_blocks

        def clean_text_keeping_structure(text):
            """Clean LaTeX text while keeping structure"""
            if not text:
                return ""

            # Remove LaTeX comments
            text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)

            # Preserve \file commands
            if '\\includegraphics' in text:
                # Convert \includegraphics to \file
                text = re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', r'\\file \1', text)

            # Remove specific LaTeX environments we'll handle separately
            text = re.sub(r'\\begin\{center\}', r'\\centering\n', text)
            text = re.sub(r'\\end\{center\}', r'', text)

            # Clean up multiple newlines
            text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

            return text.strip()

        def extract_and_clean_content(frame_content):
            """Extract and clean slide content"""
            if not frame_content:
                return []

            # First, extract TikZ blocks with their surrounding LaTeX
            content_with_placeholders, tikz_blocks = extract_complete_tikz_blocks(frame_content)

            # Extract notes separately if present
            notes_content = ""
            if '\\note{' in content_with_placeholders:
                # This is simplified - you might need better note extraction
                note_match = re.search(r'\\note\{(.*?)\}', content_with_placeholders, re.DOTALL)
                if note_match:
                    notes_content = note_match.group(1)
                    # Remove notes from main content
                    content_with_placeholders = content_with_placeholders[:note_match.start()]

            # Clean the remaining content
            cleaned_content = clean_text_keeping_structure(content_with_placeholders)

            # Restore TikZ blocks
            for i, tikz_block in enumerate(tikz_blocks):
                placeholder = f"{{TIKZ_BLOCK_{i}}}"
                cleaned_content = cleaned_content.replace(placeholder, f"\n{tikz_block}\n")

            # Split into lines and clean
            content_lines = []
            for line in cleaned_content.split('\n'):
                line = line.strip()
                if line:
                    # Handle itemize/enumerate
                    if '\\item' in line:
                        # Replace \item with -
                        line = re.sub(r'\\item\s*', '- ', line)
                        # Remove \textbf{...} but keep content
                        line = re.sub(r'\\textbf\{([^}]*)\}', r'\1', line)
                    content_lines.append(line)

            return content_lines, notes_content

        def process_notes(notes_content):
            """Process LaTeX notes into simple format"""
            notes_lines = []
            if not notes_content:
                return notes_lines

            # Simple processing - convert \item to •
            lines = notes_content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('\\item'):
                    # Clean \item command
                    line = re.sub(r'\\item\s*', '• ', line)
                    # Handle nested items
                    if '\\begin{itemize}' in line:
                        continue  # Skip begin commands
                    if '\\end{itemize}' in line:
                        continue  # Skip end commands
                    if line:
                        notes_lines.append(line)

            return notes_lines

        # Parse the TeX file
        slides = []

        # Extract document content
        doc_match = re.search(r'\\begin{document}(.*?)\\end{document}', tex_content, re.DOTALL)
        if not doc_match:
            raise ValueError("No document body found in TeX file")

        document_content = doc_match.group(1)

        # Find frames
        frame_pattern = r'\\begin\{frame\}(.*?)\\end\{frame\}'
        frame_matches = re.finditer(frame_pattern, document_content, re.DOTALL)

        slide_count = 0
        for frame_match in frame_matches:
            slide_count += 1
            frame_content = frame_match.group(1)

            # Skip title page frames
            if '\\titlepage' in frame_content or '\\maketitle' in frame_content:
                continue

            # Extract title
            title = f"Slide {slide_count}"
            title_match = re.search(r'\\frametitle\{([^}]*)\}', frame_content)
            if title_match:
                title = title_match.group(1)
            else:
                # Check for title in frame options
                title_match = re.match(r'^\s*\{([^}]+)\}', frame_content.strip())
                if title_match:
                    title = title_match.group(1)

            # Extract and clean content
            content_lines, raw_notes = extract_and_clean_content(frame_content)

            # Process notes
            notes_lines = process_notes(raw_notes)

            # Only add slides with content
            if content_lines:
                slides.append({
                    'title': title,
                    'content': content_lines,
                    'notes': notes_lines
                })

        # Write to output file in IDE format
        with open(output_path, 'w', encoding='utf-8') as f:
            for slide in slides:
                f.write(f"\\title {slide['title']}\n")
                f.write("\\begin{Content}\n")
                for content_line in slide['content']:
                    f.write(f"{content_line}\n")
                f.write("\\end{Content}\n")
                if slide['notes']:
                    f.write("\\begin{Notes}\n")
                    for note_line in slide['notes']:
                        f.write(f"{note_line}\n")
                    f.write("\\end{Notes}\n")
                f.write("\n")  # Slide separator

        return str(output_path)

    except Exception as e:
        print(f"Error converting TeX file: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_installation():
    """Silently update installed files if running from a newer version"""
    try:
        system, paths = get_installation_paths()
        current_path = Path(__file__).resolve()

        # Determine installation directory
        install_dir = paths['share'] / 'bsg-ide' if system != "Windows" else paths['bin']

        # If running from an installation directory, no need to update
        if str(current_path).startswith(str(install_dir)):
            return True

        # Get installed version info
        installed_version = "0.0.0"
        version_file = install_dir / "version.txt"
        if version_file.exists():
            installed_version = version_file.read_text().strip()

        # Compare with current version
        current_version = getattr(BeamerSlideEditor, '__version__', "2.4")

        # Always update files if versions don't match
        if installed_version != current_version:
            print(f"Updating BSG-IDE from version {installed_version} to {current_version}")

            # Create directories if they don't exist
            os.makedirs(install_dir, exist_ok=True)
            os.makedirs(paths['bin'], exist_ok=True)

            # Copy current files to installation directory
            current_dir = current_path.parent
            required_files = [
                'BSG_IDE.py',
                'BeamerSlideGenerator.py',
                'requirements.txt'
            ]

            for file in required_files:
                src = current_dir / file
                if src.exists():
                    shutil.copy2(src, install_dir)
                    print(f"Updated {file}")

            # Update version file
            version_file.write_text(current_version)

            # Update launcher script
            launcher_script = create_bsg_launcher(install_dir, paths)

            if system == "Linux":
                launcher_path = paths['bin'] / 'bsg-ide'
                launcher_path.write_text(launcher_script)
                launcher_path.chmod(0o755)

            elif system == "Windows":
                batch_content = f"""@echo off
set PYTHONPATH={install_dir};%PYTHONPATH%
pythonw "{install_dir}\\BSG_IDE.py" %*
"""
                batch_path = paths['bin'] / 'bsg-ide.bat'
                batch_path.write_text(batch_content)

            # Update desktop entry if needed
            if system == "Linux":
                desktop_entry = f"""[Desktop Entry]
Version={current_version}
Type=Application
Name=BSG-IDE
Comment=Beamer Slide Generator IDE
Exec={paths['bin']}/bsg-ide
Icon=bsg-ide
Terminal=false
Categories=Office;Development;Education;
Keywords=presentation;latex;beamer;slides;
StartupWMClass=bsg-ide
"""
                desktop_path = paths['apps'] / 'bsg-ide.desktop'
                desktop_path.write_text(desktop_entry)
                desktop_path.chmod(0o755)

            # Ensure Python paths are correct
            setup_python_paths()

            print("Update completed successfully")

        return True

    except Exception as e:
        print(f"Warning: Update check failed: {str(e)}")
        traceback.print_exc()
        return False


# Function to replace console prompts with GUI dialogs
def handle_long_institution(parent, institution_name):
    """Handle long institution name with GUI dialog"""
    dialog = InstitutionNameDialog(parent, institution_name)
    parent.wait_window(dialog)
    return dialog.short_name

def handle_media_selection(parent, title, content):
    """Handle media selection with GUI dialog"""
    dialog = MediaSelectionDialog(parent, title, content)
    parent.wait_window(dialog)
    return dialog.result
#----------------------------------------------------------Install in local bin -----------------------------------

def setup_python_paths():
    """Setup Python paths for imports"""
    import sys
    import site
    from pathlib import Path

    # Get user's home directory
    home = Path.home()

    # Add common installation paths
    paths = [
        home / '.local' / 'lib' / 'bsg-ide',  # Linux/macOS user installation
        home / '.local' / 'bin',              # Linux/macOS binaries
        home / 'Library' / 'Application Support' / 'BSG-IDE',  # macOS
        Path(site.getusersitepackages()) / 'bsg-ide',  # Windows user site-packages
    ]

    # Add installation directory to PYTHONPATH
    for path in paths:
        str_path = str(path)
        if path.exists() and str_path not in sys.path:
            sys.path.insert(0, str_path)

def create_bsg_launcher(install_dir: Path, paths: dict) -> str:
    """Create launcher script with updated terminal handling"""
    launcher_script = f"""#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import tkinter as tk
import customtkinter as ctk

# Add all possible installation paths
INSTALL_PATHS = [
    '{install_dir}',
    '{paths["bin"]}',
    str(Path.home() / '.local' / 'lib' / 'bsg-ide'),
    str(Path.home() / '.local' / 'bin'),
    str(Path.home() / 'Library' / 'Application Support' / 'BSG-IDE'),
]

# Add paths to Python path
for path in INSTALL_PATHS:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

# Import the terminal module
from BSG_terminal import InteractiveTerminal, SimpleRedirector

# Import and run main program
try:
    #from BSG_IDE import BeamerSlideEditor

    # Create application instance
    app = BeamerSlideEditor()

    # Redirect stdout and stderr to app's terminal after it's created
    if hasattr(app, 'terminal'):
        sys.stdout = SimpleRedirector(app.terminal)
        sys.stderr = SimpleRedirector(app.terminal, "red")

    # Start the application
    app.mainloop()

except Exception as e:
    import traceback
    print(f"Error starting BSG-IDE: {str(e)}")
    traceback.print_exc()
    if sys.platform != "win32":
        input("Press Enter to exit...")
"""
    return launcher_script


def setup_desktop_entry():
    """Create desktop entry and set up icons"""
    try:
        # Get user's home directory
        home_dir = Path.home()

        # Create necessary directories
        apps_dir = home_dir / '.local' / 'share' / 'applications'
        apps_dir.mkdir(parents=True, exist_ok=True)

        icon_base = home_dir / '.local' / 'share' / 'icons' / 'hicolor'

        # Copy icon to appropriate locations for different resolutions
        icon_sizes = ['16x16', '32x32', '48x48', '64x64', '128x128', '256x256']

        # First, ensure we have the source icon
        source_icon = (Path(__file__).parent / 'icons' / 'bsg-ide.png')
        if not source_icon.exists():
            source_icon = Path(__file__).parent / 'bsg-ide.png'

        if not source_icon.exists():
            print("Warning: Could not find source icon file")
            return False

        for size in icon_sizes:
            icon_dir = icon_base / size / 'apps'
            icon_dir.mkdir(parents=True, exist_ok=True)
            icon_path = icon_dir / 'bsg-ide.png'

            try:
                shutil.copy2(source_icon, icon_path)
                print(f"✓ Copied icon for size {size}")
            except Exception as e:
                print(f"Warning: Could not copy icon for size {size}: {e}")

        # Create desktop entry
        desktop_entry = """[Desktop Entry]
Version=1.0
Type=Application
Name=BSG-IDE
Comment=Beamer Slide Generator IDE
Exec=bsg-ide
Icon=bsg-ide
Terminal=false
Categories=Office;Development;Education;
Keywords=presentation;latex;beamer;slides;
StartupWMClass=bsg-ide
"""

        # Write desktop entry file
        desktop_path = apps_dir / 'bsg-ide.desktop'
        desktop_path.write_text(desktop_entry)
        desktop_path.chmod(0o755)

        # Update icon cache
        try:
            subprocess.run(['gtk-update-icon-cache', '--force', '--quiet', str(icon_base)])
            print("✓ Icon cache updated")
        except Exception as e:
            print(f"Warning: Could not update icon cache: {e}")

        print("✓ Desktop entry created successfully")
        return True

    except Exception as e:
        print(f"Error creating desktop entry: {str(e)}")
        return False

def verify_desktop_entry():
    """Verify desktop entry and icon installation"""
    try:
        home_dir = Path.home()
        desktop_entry = home_dir / '.local' / 'share' / 'applications' / 'bsg-ide.desktop'
        icon_path = home_dir / '.local' / 'share' / 'icons' / 'hicolor' / '128x128' / 'apps' / 'bsg-ide.png'

        if not desktop_entry.exists() or not icon_path.exists():
            print("Desktop entry or icon missing. Setting up...")
            return setup_desktop_entry()

        return True

    except Exception as e:
        print(f"Error verifying desktop entry: {str(e)}")
        return False



#-------------------------------------------------pympress installation -----------------------------
def setup_pympress():
    """Verify pympress installation using system packages"""
    try:
        # Check if pympress works by importing required modules
        def check_pympress_deps():
            try:
                import gi
                import cairo
                gi.require_version('Gtk', '3.0')
                from gi.repository import Gtk
                import pympress
                return True
            except ImportError:
                return False

        if check_pympress_deps():
            print("✓ Pympress and dependencies already installed")
            return True

        print("Installing pympress and dependencies using pip...")

        # Install using pip (system level)
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--user", "pympress"
            ])
            print("✓ pympress installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install pympress: {e}")
            return False

    except Exception as e:
        print(f"✗ Error setting up pympress: {str(e)}")
        traceback.print_exc()
        return False

def launch_pympress(pdf_path):
    """Launch pympress with proper environment setup"""
    try:
        # Ensure absolute path
        abs_pdf_path = os.path.abspath(pdf_path)

        # Set up environment variables
        env = os.environ.copy()

        # Add GI typelib path if needed
        typelib_paths = [
            '/usr/lib/girepository-1.0',
            '/usr/local/lib/girepository-1.0',
            '/usr/lib/x86_64-linux-gnu/girepository-1.0'
        ]

        existing_typelib = env.get('GI_TYPELIB_PATH', '').split(':')
        new_typelib = ':'.join(filter(os.path.exists, typelib_paths + existing_typelib))
        env['GI_TYPELIB_PATH'] = new_typelib

        # Try to launch pympress
        if sys.platform.startswith('win'):
            subprocess.Popen(['pympress', abs_pdf_path], shell=True, env=env)
        else:
            # Try different possible pympress locations
            pympress_paths = [
                shutil.which('pympress'),
                '/usr/local/bin/pympress',
                '/usr/bin/pympress',
                os.path.expanduser('~/.local/bin/pympress')
            ]

            for path in filter(None, pympress_paths):
                if os.path.exists(path):
                    subprocess.Popen([path, abs_pdf_path], env=env)
                    return True

            raise FileNotFoundError("pympress executable not found")

    except Exception as e:
        print(f"Error launching pympress: {str(e)}")
        traceback.print_exc()
        return False
#-------------------------------------------------------------------------
def create_icon(install_dir: Path) -> bool:
    """Create icon with airis4D logo and BSG-IDE text below"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import os

        # Create icons directory
        icons_dir = install_dir / 'icons'
        os.makedirs(icons_dir, exist_ok=True)

        # Icon sizes needed for different platforms
        sizes = [16, 32, 48, 64, 128, 256]

        # Create base icon image (make it square)
        size = 256  # Base size
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw airis4D logo - maintaining exact proportions from ASCII art
        logo_height = size * 0.6  # Logo takes up 60% of height
        margin = size * 0.2  # 20% margin from top

        # Calculate logo dimensions
        logo_width = logo_height * 0.8  # Maintain aspect ratio

        # Logo starting position
        start_x = (size - logo_width) / 2
        start_y = margin

        # Draw the triangle (airis4D logo)
        triangle_points = [
            (start_x, start_y + logo_height),  # Bottom left
            (start_x + logo_width/2, start_y), # Top
            (start_x + logo_width, start_y + logo_height)  # Bottom right
        ]

        # Draw outer triangle
        draw.polygon(triangle_points, fill=(255, 0, 0, 255))  # Red for outer triangle

        # Calculate inner triangle points (80% size of outer)
        inner_scale = 0.8
        inner_offset_x = (logo_width * (1 - inner_scale)) / 2
        inner_offset_y = (logo_height * (1 - inner_scale))
        inner_points = [
            (start_x + inner_offset_x, start_y + logo_height - inner_offset_y),
            (start_x + logo_width/2, start_y + inner_offset_y),
            (start_x + logo_width - inner_offset_x, start_y + logo_height - inner_offset_y)
        ]
        draw.polygon(inner_points, fill=(0, 0, 0, 255))  # Black for inner triangle

        # Add "BSG-IDE" text below logo
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(size * 0.15))
        except:
            font = ImageFont.load_default()

        text = "BSG-IDE"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (size - text_width) // 2
        text_y = start_y + logo_height + (size * 0.05)  # Small gap between logo and text
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0, 255))

        # Save in different sizes
        for icon_size in sizes:
            resized = img.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
            icon_path = icons_dir / f'bsg-ide_{icon_size}x{icon_size}.png'
            resized.save(icon_path, 'PNG')

            # For Linux, also save in appropriate hicolor directory
            if sys.platform.startswith('linux'):
                hicolor_path = Path.home() / '.local' / 'share' / 'icons' / 'hicolor' / f'{icon_size}x{icon_size}' / 'apps'
                os.makedirs(hicolor_path, exist_ok=True)
                resized.save(hicolor_path / 'bsg-ide.png', 'PNG')

        # Create .ico file for Windows
        if sys.platform.startswith('win'):
            icon_sizes = [(16,16), (32,32), (48,48), (256,256)]
            images = []
            for size in icon_sizes:
                resized = img.resize(size, Image.Resampling.LANCZOS)
                images.append(resized)
            ico_path = icons_dir / 'bsg-ide.ico'
            img.save(ico_path, format='ICO', sizes=icon_sizes)

        return True

    except Exception as e:
        print(f"Error creating icon: {str(e)}")
        traceback.print_exc()
        return False
#--------------------------------------------------------------------------
def setup_static_directory():
    """
    Create required static directory structure for PyMuPDF
    """
    print("Setting up static directory structure...")

    # Create required directories
    directories = [
        'static',
        'static/css',
        'static/js',
        'static/images'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

    # Create minimal required files
    minimal_files = {
        'static/css/style.css': '/* Minimal CSS */',
        'static/js/script.js': '// Minimal JS',
        'static/.keep': ''  # Empty file to ensure directory is tracked
    }

    for filepath, content in minimal_files.items():
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Created file: {filepath}")



def check_windows_dependencies() -> None:
    """
    Check and setup dependencies for Windows.
    """
    # Check for MiKTeX or TeX Live
    if not (shutil.which('pdflatex') or os.path.exists(r'C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe')):
        print("\nLaTeX is not installed. Please install MiKTeX:")
        print("1. Visit: https://miktex.org/download")
        print("2. Download and install MiKTeX")
        print("3. Run this script again")
        sys.exit(1)

def check_linux_dependencies(system_commands: List[Tuple[str, str]]) -> None:
    """
    Check and setup dependencies for Linux.
    """
    missing_packages = []

    for cmd, packages in system_commands:
        if not shutil.which(cmd):
            missing_packages.append(packages)

    if missing_packages:
        print("\nSome system packages are missing. Installing...")
        try:
            # Try to detect package manager
            if shutil.which('apt'):
                install_cmd = ['sudo', 'apt', 'install', '-y']
            elif shutil.which('dnf'):
                install_cmd = ['sudo', 'dnf', 'install', '-y']
            elif shutil.which('pacman'):
                install_cmd = ['sudo', 'pacman', '-S', '--noconfirm']
            else:
                print("Could not detect package manager. Please install manually:")
                print(" ".join(missing_packages))
                sys.exit(1)

            for packages in missing_packages:
                subprocess.check_call(install_cmd + packages.split())
                print(f"✓ Installed {packages}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing system packages: {str(e)}")
            sys.exit(1)

def check_bsg_file() -> None:
    """
    Check for BeamerSlideGenerator.py and download if missing.
    """
    if not os.path.exists('BeamerSlideGenerator.py'):
        print("\nBeamerSlideGenerator.py not found. Downloading...")
        try:
            import requests
            # Replace with actual URL to your BeamerSlideGenerator.py
            url = "https://raw.githubusercontent.com/sajeethphilip/BeamerSlideGenerator/main/BeamerSlideGenerator.py"
            response = requests.get(url)
            response.raise_for_status()

            with open('BeamerSlideGenerator.py', 'w') as f:
                f.write(response.text)
            print("✓ BeamerSlideGenerator.py downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading BeamerSlideGenerator.py: {str(e)}")
            print("\nPlease manually download BeamerSlideGenerator.py and place it in the same directory.")
            sys.exit(1)

def create_footer(self) -> None:
    """Create footer with institution info and links"""
    # Footer frame with dark theme
    self.footer = ctk.CTkFrame(self)
    self.footer.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

    # Left side - Institution name
    inst_label = ctk.CTkLabel(
        self.footer,
        text="Artificial Intelligence Research and Intelligent Systems (airis4D)",
        font=("Arial", 12, "bold"),
        text_color="#4ECDC4"  # Using the same color scheme as the editor
    )
    inst_label.pack(side="left", padx=10)

    # Right side - Contact and GitHub links
    links_frame = ctk.CTkFrame(self.footer, fg_color="transparent")
    links_frame.pack(side="right", padx=10)

    # Contact link
    contact_button = ctk.CTkButton(
        links_frame,
        text="nsp@airis4d.com",
        command=lambda: webbrowser.open("mailto:nsp@airis4d.com"),
        fg_color="transparent",
        text_color="#FFB86C",  # Using the bracket color from syntax highlighting
        hover_color="#2F3542",
        height=20
    )
    contact_button.pack(side="left", padx=5)

    # Separator
    separator = ctk.CTkLabel(
        links_frame,
        text="|",
        text_color="#6272A4"  # Using comment color from syntax highlighting
    )
    separator.pack(side="left", padx=5)

    # GitHub link with small icon
    github_button = ctk.CTkButton(
        links_frame,
        text="GitHub",
        command=lambda: webbrowser.open("https://github.com/sajeethphilip/Beamer-Slide-Generator.git"),
        fg_color="transparent",
        text_color="#FFB86C",
        hover_color="#2F3542",
        height=20
    )
    github_button.pack(side="left", padx=5)

    # License info
    license_label = ctk.CTkLabel(
        links_frame,
        text="(Creative Commons License)",
        font=("Arial", 10),
        text_color="#6272A4"
    )
    license_label.pack(side="left", padx=5)

# Import BeamerSlideGenerator functions
try:
    from BeamerSlideGenerator import (
        get_beamer_preamble,
        process_media,
        generate_latex_code,
        download_youtube_video,
        construct_search_query,
        open_google_image_search
    )
except ImportError:
    print("Error: BeamerSlideGenerator.py must be in the same directory.")
    sys.exit(1)



def get_installation_paths():
    """Get platform-specific installation paths"""
    import platform
    import os
    from pathlib import Path
    import site
    import sys

    system = platform.system()
    paths = {}

    if os.geteuid() == 0:  # Running as root/sudo
        if system == "Linux":
            paths.update({
                'bin': Path('/usr/local/bin'),
                'share': Path('/usr/local/share'),
                'icons': Path('/usr/share/icons/hicolor'),
                'apps': Path('/usr/share/applications')
            })
        else:
            # For other systems when running as root/admin
            paths.update({
                'bin': Path(sys.prefix) / 'bin',
                'share': Path(sys.prefix) / 'share'
            })
    else:  # Running as normal user
        if system == "Linux":
            # Get user's local bin from PATH or create in ~/.local/bin
            user_bin = None
            for path in os.environ.get('PATH', '').split(os.pathsep):
                if '/.local/bin' in path and os.access(path, os.W_OK):
                    user_bin = Path(path)
                    break
            if not user_bin:
                user_bin = Path.home() / '.local' / 'bin'

            paths.update({
                'bin': user_bin,
                'share': Path.home() / '.local' / 'share',
                'icons': Path.home() / '.local' / 'share' / 'icons' / 'hicolor',
                'apps': Path.home() / '.local' / 'share' / 'applications'
            })

        elif system == "Windows":
            appdata = Path(os.getenv('APPDATA'))
            paths.update({
                'bin': appdata / 'BSG-IDE',
                'shortcut': appdata / 'Microsoft' / 'Windows' / 'Start Menu' / 'Programs' / 'BSG-IDE'
            })

        elif system == "Darwin":  # macOS
            paths.update({
                'app': Path.home() / 'Applications' / 'BSG-IDE.app',
                'contents': Path.home() / 'Applications' / 'BSG-IDE.app' / 'Contents',
                'bin': Path.home() / 'Applications' / 'BSG-IDE.app' / 'Contents' / 'MacOS',
                'resources': Path.home() / 'Applications' / 'BSG-IDE.app' / 'Contents' / 'Resources'
            })

    return system, paths

def check_installation():
    """Check if BSG-IDE is installed in the system"""
    system, paths = get_installation_paths()

    if system == "Linux":
        return (paths['bin'] / 'bsg-ide').exists()
    elif system == "Windows":
        return (paths['bin'] / 'bsg-ide.pyw').exists()
    elif system == "Darwin":
        return paths['app'].exists()
    return False
def make_executable(file_path):
    """Make file executable on Unix systems or create launcher on Windows"""
    import stat
    import platform

    system = platform.system()
    if system != "Windows":
        # Add executable permission for owner
        current = stat.S_IMODE(os.lstat(file_path).st_mode)
        os.chmod(file_path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    else:
        # For Windows, create a .bat launcher
        bat_path = file_path.parent / 'bsg-ide.bat'
        with open(bat_path, 'w') as f:
            f.write(f'@echo off\n"{sys.executable}" "{file_path}" %*')

def check_bsg_file():
    """
    Check for BeamerSlideGenerator.py and install if missing.
    """
    # Get the correct installation path based on platform
    import platform
    system = platform.system()

    if system == "Linux":
        install_lib = Path.home() / '.local' / 'lib' / 'bsg-ide'
    elif system == "Windows":
        install_lib = Path(os.getenv('APPDATA')) / 'BSG-IDE'
    else:  # macOS
        install_lib = Path.home() / 'Library' / 'Application Support' / 'BSG-IDE'

    install_lib.mkdir(parents=True, exist_ok=True)
    bsg_file = install_lib / 'BeamerSlideGenerator.py'

    if not bsg_file.exists():
        print("\nBeamerSlideGenerator.py not found in installation directory. Installing...")
        try:
            # First check if it exists in current directory
            current_bsg = Path('BeamerSlideGenerator.py')
            if current_bsg.exists():
                # Copy the file to installation directory
                shutil.copy2(current_bsg, bsg_file)
                print(f"✓ BeamerSlideGenerator.py installed to {bsg_file}")
            else:
                # Look in script's directory
                script_dir = Path(__file__).parent.resolve()
                script_bsg = script_dir / 'BeamerSlideGenerator.py'
                if script_dir.exists():
                    shutil.copy2(script_bsg, bsg_file)
                    print(f"✓ BeamerSlideGenerator.py installed to {bsg_file}")
                else:
                    print("✗ Error: BeamerSlideGenerator.py not found in current or script directory.")
                    print("Please ensure BeamerSlideGenerator.py is in the same directory as BSG_IDE.py")
                    sys.exit(1)
        except Exception as e:
            print(f"✗ Error installing BeamerSlideGenerator.py: {str(e)}")
            print("\nPlease manually copy BeamerSlideGenerator.py to:", bsg_file)
            sys.exit(1)
    else:
        print("✓ BeamerSlideGenerator.py is installed")

    # Also ensure the file is in the current working directory for direct script mode
    if not Path('BeamerSlideGenerator.py').exists():
        try:
            shutil.copy2(bsg_file, 'BeamerSlideGenerator.py')
        except Exception as e:
            print(f"Warning: Could not copy BeamerSlideGenerator.py to current directory: {e}")


#-------------------------------------INSTALLATION----------------------------
import os
import sys
import platform
import shutil
import subprocess
import json
from pathlib import Path
import site
import socket
import ctypes
from importlib import util

def main():
    """Main entry point for both source and installed runs"""
    try:
        # Setup paths first
        package_root, resources_dir = setup_paths()

        # Parse arguments
        import argparse
        parser = argparse.ArgumentParser(description='BSG-IDE - Beamer Slide Generator IDE')
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--fix', action='store_true',
                          help='Fix installation issues')
        group.add_argument('--install', action='store_true',
                          help='Install BSG-IDE')
        group.add_argument('--verify', action='store_true',
                          help='Verify installation')

        args = parser.parse_args()

        if args.fix or args.install:
            #from installation import install_bsg_ide
            success = install_bsg_ide(fix_mode=args.fix)
            verify_installation()
            sys.exit(0 if success else 1)

        # Launch IDE

        app = BeamerSlideEditor()
        app.mainloop()

    except Exception as e:
        print(f"Error starting BSG-IDE: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
#---------------------------------------------------------------------------------------
