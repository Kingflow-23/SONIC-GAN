from PIL import Image
import os

# Load the image
base_path = r"games/sonic/sprites"
sprite_sheet = Image.open(base_path + "\Sonic.png")

"""# Define the size of each sprite block
# Define the size of each sprite block (16x16) and the spacing between them
# Define the size of each sprite block and the spacing between them
sprite_width, sprite_height = 16, 16
grid_spacing = 1  # 1-pixel spacing between each sprite

# Calculate the number of sprites in rows and columns, considering the spacing
sheet_width, sheet_height = sprite_sheet.size
num_cols = (sheet_width + grid_spacing) // (sprite_width + grid_spacing)
num_rows = (sheet_height + grid_spacing) // (sprite_height + grid_spacing)

# Create a folder to save the cropped sprites
output_folder = base_path + r'\cropped_sprites'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through each sprite block, including any edge blocks
sprite_count = 0
for row in range(num_rows):
    for col in range(num_cols):
        # Calculate the position of each sprite, skipping the grid lines
        left = col * (sprite_width + grid_spacing)
        upper = row * (sprite_height + grid_spacing)
        right = left + sprite_width
        lower = upper + sprite_height

        # Ensure we don't crop outside the image bounds, capturing all sprites including partial rows
        if left < sheet_width and upper < sheet_height:
            right = min(right, sheet_width)
            lower = min(lower, sheet_height)

            # Crop and save the sprite
            sprite = sprite_sheet.crop((left, upper, right, lower))
            sprite_path = f"{output_folder}/sprite_{sprite_count}.png"
            sprite.save(sprite_path)
            sprite_count += 1
            print(f"Cropped and saved: {sprite_path}")

print(f"Total sprites cropped: {sprite_count}")"""

# Reload the Sonic sprite sheet
sonic_sheet = Image.open(sprite_sheet)

# Define new dimensions for each sprite (40x40 as specified)
sprite_width = 40
sprite_height = 40

sheet_width, sheet_height = sprite_sheet.width, sprite_sheet.height
# Recalculate the number of rows and columns based on the new dimensions
columns = sheet_width // sprite_width
rows = sheet_height // sprite_height

# Create a list to store cropped Sonic sprites (40x40)
cropped_sonic_sprites_40x40 = []

# Iterate through the sprite sheet and crop each Sonic position
for row in range(rows):
    for col in range(columns):
        left = col * sprite_width
        upper = row * sprite_height
        right = left + sprite_width
        lower = upper + sprite_height

        # Crop the sprite
        sprite = sonic_sheet.crop((left, upper, right, lower))
        cropped_sonic_sprites_40x40.append(sprite)

# Save cropped Sonic sprites to files for verification
output_sonic_paths_40x40 = []
for index, sprite in enumerate(cropped_sonic_sprites_40x40):
    output_path = f"/mnt/data/sonic_40x40_{index + 1}.png"
    sprite.save(output_path)
    output_sonic_paths_40x40.append(output_path)

output_sonic_paths_40x40
