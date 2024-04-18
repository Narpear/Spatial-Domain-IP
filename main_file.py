import cv2

for i in range (1, 150):
    # Read the image from disk
    img = cv2.imread(fr"Corrupt Frames\frame_-{i:03d}.png", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(fr"Uncorrupt Frames\frame_-{i:03d}.png", img)

for i in range (1, 101):
    # Read the image from disk
    img = cv2.imread(fr"Corrupt Frames\frame_0{i:03d}.png", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(fr"Uncorrupt Frames\frame_0{i:03d}.png", img)


for i in range (101, 226):
    # Read the image from disk
    img = cv2.imread(fr"Corrupt Frames\frame_0{i}.png", cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded correctly
    if img is None:
        print("Could not open or find the image")
    else:
        inverted = 255 - img

        if inverted is None:
            print("None")
        else:
            cv2.imwrite(fr"Uncorrupt Frames\frame_0{i}_inverted.png", inverted)
            # print("Done")

        # cv2.imshow("image", inverted)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # print(x)
            

import numpy as np

for i in range (226, 351):
    # Read the image from disk
    image = cv2.imread(fr"Corrupt Frames\frame_0{i}.png", cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded correctly
    if image is None:
        print("Could not open or find the image")
    else:
        filtered_image = cv2.medianBlur(image, 9)
        laplacian = cv2.Laplacian(filtered_image, cv2.CV_64F)
        sharpened_image = np.uint8(np.clip(filtered_image - 0.4*laplacian, 0, 255))

        if sharpened_image is None:
            print("None")
        else:
            cv2.imwrite(fr"Uncorrupt Frames\frame_0{i}_filtered.png", sharpened_image)
            # print("Done")

        # cv2.imshow("image", inverted)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # print(x)
            


import numpy as np
import cv2

for i in range (351, 476):
    # Read the image from disk
    image = cv2.imread(fr"Corrupt Frames\frame_0{i}.png", cv2.IMREAD_COLOR)

    # Check if the image was loaded correctly
    if image is None:
        print("Could not open or find the image")
    else:
        # Convert image to LAB color space
        img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split the LAB image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(img_lab)

        # Apply CLAHE to all three channels
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        a_channel = clahe.apply(a_channel)
        b_channel = clahe.apply(b_channel)

        # Merge the processed channels back together
        img_lab_eq = cv2.merge((l_channel, a_channel, b_channel))

        # Convert the LAB image back to BGR color space
        img_output = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)   

        laplacian = cv2.Laplacian(img_output, cv2.CV_64F)
        sharpened_image = np.uint8(np.clip(img_output - 0.4*laplacian, 0, 255))
        
        img_output = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY) 

        if img_output is None:
            print("None")
        else:
            cv2.imwrite(fr"Uncorrupt Frames\frame_0{i}_gradient.png", img_output)
            # print("Done")


import cv2
import numpy as np

for i in range (476, 601):
    # Read the image from disk
    img = cv2.imread(fr"Corrupt Frames\frame_0{i}.png", cv2.IMREAD_COLOR)

    # Check if the image was loaded correctly
    if img is None:
        print("Could not open or find the image")
    else:

        denoised_image = cv2.GaussianBlur(img, (15, 15), 0)

        # # Apply Laplacian filter
        laplacian = cv2.Laplacian(denoised_image, cv2.CV_64F)

        # Convert the output back to uint8
        sharpened_image = np.uint8(np.clip(denoised_image - 0.6*laplacian, 0, 255))

        if sharpened_image is None:
            print("None")
        else:
            cv2.imwrite(fr"Uncorrupt Frames\frame_0{i}_inverted.png", sharpened_image)
            
for i in range (601, 726):
    # Read the image from disk
    img = cv2.imread(fr"Corrupt Frames\frame_0{i}.png", cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded correctly
    if img is None:
        print("Could not open or find the image")
    else:
    
        # Create a mask that has the most significant bit set
        mask = np.full((img.shape[0], img.shape[1]),  8, dtype=np.uint8)

        # Apply the mask to the image using bitwise AND
        bit_plane_4 = cv2.bitwise_and(img, mask)

        # Scale the result for better visibility
        bit_plane_4 = bit_plane_4 *  255

        denoised_image = cv2.GaussianBlur(bit_plane_4, (13, 13), 0)

        cv2.imwrite(fr"Uncorrupt Frames\frame_0{i}_inverted.png", denoised_image)


    # print(x)