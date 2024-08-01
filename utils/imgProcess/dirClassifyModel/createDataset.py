from PIL import Image
import os

# Step 1: Load the image
for i in range (21,26):
    image_path = f"C:/Users/happyelements/Desktop/dataset/orig_images/{i}.png"  # Replace with your image path
    original_image = Image.open(image_path)

    # Create output directory if it doesn't exist
    output_dir = f"{i}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 2: Rotate the image and save the central 40x40 pixels
    for angle in range(0,-360,-1):
        rotated_image = original_image.rotate(angle, resample=Image.BICUBIC)

        # Step 3: Crop the central 40x40 pixels
        width, height = rotated_image.size
        left = (width - 40) / 2
        top = (height - 40) / 2
        right = (width + 40) / 2
        bottom = (height + 40) / 2
        cropped_image = rotated_image.crop((left, top, right, bottom))

        # Step 4: Classify the image based on the angle
        if -360 <= angle < -345 or -15 <= angle <= 0:
            category = 12
        else:
            category = (abs(angle) - 15) // 30 + 1

        # Save the cropped image
        output_path = os.path.join(output_dir, f"angle_{abs(angle)}_category_{category}.png")
        cropped_image.save(output_path)

    print(f"{i} Images have been processed and saved.")
