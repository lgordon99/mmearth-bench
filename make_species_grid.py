from io import BytesIO
from pathlib import Path
from PIL import Image
import json
import requests
import time
import base64
import html
import utils

data_dir_path = utils.read_yaml('config-user.yml')['data_dir_path']

with open(f'{data_dir_path}/species/species_labels.json', 'r') as f:
    species_labels = json.load(f)

sorted_species = sorted(species_labels.items(), key=lambda x: x[1])
species_image_dir = Path(f'{data_dir_path}/species/species_images')
species_image_dir.mkdir(exist_ok=True)

def download_species_image(species_name, label_num):
    """Download and save species image from iNaturalist."""

    def find_exact_match(url):
        """Search iNaturalist API and return exact name match if found."""
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()

            if 'results' in data and len(data['results']) > 0:
                species_lower = species_name.strip().lower()

                for result in data['results']:
                    if result.get('name', '').strip().lower() == species_lower:
                        return result

        return None

    # Try multiple search strategies to find inactive taxa
    encoded_name = species_name.replace(' ', '%20')
    search_urls = [f"https://api.inaturalist.org/v1/taxa?q={encoded_name}",
                   f"https://api.inaturalist.org/v1/taxa?q={encoded_name}&is_active=false",  # Explicitly search for inactive
                   f"https://api.inaturalist.org/v1/taxa?q={encoded_name}&all_names=true",  # Include all names
                   f"https://api.inaturalist.org/v1/taxa/autocomplete?q={encoded_name}",
                   f"https://api.inaturalist.org/v1/taxa?q=\"{encoded_name}\""]

    taxon = None

    for url in search_urls:
        taxon = find_exact_match(url)

        if taxon:
            break

    # Download and save image
    img_response = requests.get(taxon['default_photo']['medium_url'], timeout=10)

    if img_response.status_code == 200:
        img = Image.open(BytesIO(img_response.content))

        if img.mode in ('P', 'RGBA', 'LA'):
            img = img.convert('RGB')

        img.save(species_image_dir / f"{label_num:02d}_{species_name.replace(' ', '_')}.jpg", 'JPEG', quality=85)
        status = " (inactive)" if not taxon.get('is_active', True) else ""
        print(f"  {species_name}: Downloaded from iNaturalist{status}")

# Download all images
def download_species_images():
    print("Downloading images for 100 species...")

    for idx, (species_name, label_num) in enumerate(sorted_species):
        print(f"[{idx+1}/100] {species_name}")
        download_species_image(species_name, label_num)
        time.sleep(0.2) # rate limiting

    print("\nDownload complete! Now run the grid creation script.")

def create_species_grid():
    """Create a 10x10 grid of species images with labels underneath as SVG."""
    # Load all images and sort by label number
    image_files = sorted(species_image_dir.glob("*.jpg"), key=lambda x: int(x.stem.split('_')[0]))

    # Image settings
    cell_size = 400  # Size of each cell (image + label)
    image_size = 360  # Size of the actual image
    label_height = 40  # Height for text label
    grid_size = 10
    padding = 10

    # Calculate grid dimensions
    grid_width = grid_size * cell_size + (grid_size - 1) * padding
    grid_height = grid_size * cell_size + (grid_size - 1) * padding

    # Corner radius for rounded images
    corner_radius = 10

    # Start building SVG
    svg_content = []
    svg_content.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {grid_width} {grid_height}" width="{grid_width}" height="{grid_height}">')
    svg_content.append('<defs>')
    svg_content.append('  <style>')
    svg_content.append('    .species-label { font-family: Arial, sans-serif; font-size: 28px; fill: white; text-anchor: middle; }')
    svg_content.append('  </style>')

    # Pre-calculate all image positions for clipPaths
    image_positions = []
    for idx, img_file in enumerate(image_files[:100]):  # Limit to 100
        row = idx // grid_size
        col = idx % grid_size
        x = col * (cell_size + padding)
        y = row * (cell_size + padding)
        img_x = x + (cell_size - image_size) // 2
        img_y = y + (cell_size - image_size - label_height) // 2
        image_positions.append((idx, img_x, img_y))

    # Add clipPaths to defs
    for idx, img_x, img_y in image_positions:
        clip_id = f'clip-{idx}'
        svg_content.append(f'  <clipPath id="{clip_id}">')
        svg_content.append(f'    <rect x="{img_x}" y="{img_y}" width="{image_size}" height="{image_size}" rx="{corner_radius}" ry="{corner_radius}"/>')
        svg_content.append(f'  </clipPath>')

    svg_content.append('</defs>')

    # Place images in grid
    for idx, img_file in enumerate(image_files[:100]):  # Limit to 100
        row = idx // grid_size
        col = idx % grid_size

        # Calculate position
        x = col * (cell_size + padding)
        y = row * (cell_size + padding)

        # Load and center-crop image to fixed size
        img = Image.open(img_file)

        # Calculate scaling to cover the target size
        width, height = img.size
        target_size = image_size

        # Scale to cover (maintain aspect ratio)
        scale = max(target_size / width, target_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop to exact target size
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        img = img.crop((left, top, right, bottom))

        # Convert image to base64
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        img_data_uri = f"data:image/png;base64,{img_base64}"

        # Get image position (already calculated)
        img_x, img_y = image_positions[idx][1], image_positions[idx][2]
        clip_id = f'clip-{idx}'

        # Add image to SVG with rounded corners
        svg_content.append(f'  <image x="{img_x}" y="{img_y}" width="{image_size}" height="{image_size}" href="{img_data_uri}" clip-path="url(#{clip_id})"/>')

        # Extract species name from filename
        species_name = img_file.stem.split('_', 1)[1].replace('_', ' ')
        species_name_escaped = html.escape(species_name)

        # Calculate text position (centered)
        text_x = x + cell_size // 2
        text_y = y + image_size + 30

        # Add text label to SVG
        svg_content.append(f'  <text x="{text_x}" y="{text_y}" class="species-label">{species_name_escaped}</text>')

    # Close SVG
    svg_content.append('</svg>')

    # Save the SVG
    output_path = species_image_dir.parent / 'species_grid.svg'
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_content))

    print(f"\nSpecies grid saved to: {output_path}")

if __name__ == "__main__":
    download_species_images()
    create_species_grid()
