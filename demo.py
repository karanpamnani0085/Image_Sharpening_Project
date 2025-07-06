import torch
import cv2
import os
import numpy as np
from model.student import StudentModel
from utils.tools import get_device, load_checkpoint
import torchvision.transforms as transforms

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def process_frame(frame, model, transform, device):
    # Use a larger size for sharper results (match your training size)
    input_size = (320, 320)  # Change to your training size!
    resized_frame = cv2.resize(frame, input_size)
    frame_tensor = transform(resized_frame).unsqueeze(0).to(device, non_blocking=True)
    if next(model.parameters()).dtype == torch.half:
        frame_tensor = frame_tensor.half()
    with torch.no_grad():
        sharpened = model(frame_tensor)
    sharpened = sharpened.squeeze(0).cpu()
    sharpened = denormalize(sharpened, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    sharpened = torch.clamp(sharpened, 0, 1)
    sharpened = (sharpened.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    # Resize back to original frame size for display
    sharpened = cv2.resize(sharpened, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return sharpened

def main():
    device = get_device()
    model = StudentModel(features=32).to(device)  # Use the same features as your checkpoint
    if device.type == 'cuda':
        model = model.half()
        torch.backends.cudnn.benchmark = True
    model.eval()

    # Always use the 50th checkpoint
    checkpoint_path = os.path.join('checkpoints', 'student_epoch_50.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    print(f"Loading checkpoint: {checkpoint_path}")
    load_checkpoint(checkpoint_path, model)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame at training size, then resize back for display
        sharpened = process_frame(frame_rgb, model, transform, device)
        sharpened_bgr = cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR)

        # Stack and display
        display = np.hstack([frame, sharpened_bgr])
        cv2.imshow('Sharpened vs Original', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()