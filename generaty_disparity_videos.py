import os, glob, cv2
import torch, numpy as np
from tqdm import tqdm

# ── 설정 ──
ROOT_DIR   = "/data/holoassist"
PATTERN    = os.path.join(ROOT_DIR, "*", "Export_py", "Video_compress.mp4")
BATCH_SIZE = 84
USE_HALF   = True
DOWNSAMPLE = 1
DEVICE     = "cuda:0"

# ── 모델 & transform 초기화 ──
device = torch.device(DEVICE)
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
if USE_HALF:
    midas = midas.half()
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# default_transform은 [1,3,H,W]를 리턴하므로, 밑에서 squeeze 해줄 겁니다.
midas_transform = transforms.default_transform

def depth2disparity(xd, dmin=0.1, dmax=10.0):
    xdisp      = 1.0 / np.sqrt(np.clip(xd, dmin, dmax))
    xdisp_norm = xdisp / (xdisp.max() + 1e-6) * 2 - 1
    return np.stack([xdisp_norm]*3, axis=0).astype(np.float32)

def process_video(in_path):
    out_path = os.path.join(os.path.dirname(in_path), "Disparity_compress.mp4")
    cap = cv2.VideoCapture(in_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30
        print(f"⚠️ FPS가 0으로 읽혀 {fps}로 강제 설정합니다.")
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * DOWNSAMPLE)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * DOWNSAMPLE)
    
    # 안정적인 mp4 코덱 선택
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter open failed: fourcc={fourcc}, fps={fps}, size={(w,h)}")
    
    # 프레임 읽기
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if DOWNSAMPLE != 1:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    
    # 배치 처리
    for i in tqdm(range(0, len(frames), BATCH_SIZE), desc=os.path.basename(in_path)):
        batch = frames[i : i + BATCH_SIZE]
        tsr_list = []
        for f in batch:
            img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t   = midas_transform(img)
            if t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            tsr_list.append(t)
        tsr = torch.stack(tsr_list, dim=0).to(device)
        if USE_HALF:
            tsr = tsr.half()

        with torch.no_grad():
            depth_batch = midas(tsr)
        if depth_batch.ndim == 4:
            depth_batch = depth_batch.squeeze(1)

        for depth_map in depth_batch.cpu().numpy():
            disp = depth2disparity(depth_map)
            vis  = ((np.transpose(disp, (1,2,0)) + 1)*0.5*255).astype(np.uint8)
            vis_resized = cv2.resize(vis, (w, h), interpolation=cv2.INTER_LINEAR)
            writer.write(cv2.cvtColor(vis_resized, cv2.COLOR_RGB2BGR))

    writer.release()
    print("✅ Saved:", out_path)
def main():
    vids = sorted(glob.glob(PATTERN))
    if not vids:
        print("No videos found.")
        return
    
    for v in tqdm(vids):
        process_video(v)

if __name__=="__main__":
    main()
