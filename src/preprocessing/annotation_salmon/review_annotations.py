#!/usr/bin/env python3
"""
交互式标注审核工具 - 快速标记误检图像

使用方法：
    python review_annotations.py --visualized data/visualized/salmon_validation/

键盘操作：
    → 或 Space : 下一张（保留）
    ← : 上一张
    d 或 Delete : 标记为删除
    u : 取消标记（撤销）
    q : 退出并保存标记列表
"""

import cv2
import argparse
from pathlib import Path
import json


class AnnotationReviewer:
    def __init__(self, visualized_dir: Path):
        self.visualized_dir = visualized_dir
        self.images = sorted(list(visualized_dir.glob("*.jpg")))
        self.current_idx = 0
        self.marked_for_deletion = set()
        self.window_name = "标注审核 [→/Space=下一张 | ←=上一张 | D=删除 | U=撤销 | Q=退出]"
        
        print(f"\n找到 {len(self.images)} 张图像")
        print(f"\n键盘操作:")
        print(f"  → 或 Space : 下一张（保留）")
        print(f"  ← : 上一张")
        print(f"  D 或 Delete : 标记为删除")
        print(f"  U : 取消标记（撤销）")
        print(f"  Q : 退出并保存")
        print(f"\n开始审核...\n")
    
    def show_current_image(self):
        if self.current_idx >= len(self.images):
            print("\n✅ 已审核完所有图像！")
            return False
        
        img_path = self.images[self.current_idx]
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"❌ 无法读取: {img_path.name}")
            self.current_idx += 1
            return True
        
        # 调整图像大小适应屏幕
        h, w = img.shape[:2]
        max_h, max_w = 1000, 1800
        if h > max_h or w > max_w:
            scale = min(max_h/h, max_w/w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        # 添加状态信息
        status_text = f"[{self.current_idx + 1}/{len(self.images)}] {img_path.name}"
        if img_path in self.marked_for_deletion:
            status_text += " [标记删除]"
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)
        
        cv2.putText(img, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"已标记删除: {len(self.marked_for_deletion)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow(self.window_name, img)
        return True
    
    def run(self):
        while self.current_idx < len(self.images):
            if not self.show_current_image():
                break
            
            key = cv2.waitKey(0) & 0xFF
            
            # 下一张
            if key == 83 or key == ord(' ') or key == ord('n'):  # → or Space
                self.current_idx += 1
            
            # 上一张
            elif key == 81 or key == ord('p'):  # ←
                if self.current_idx > 0:
                    self.current_idx -= 1
            
            # 标记删除
            elif key == ord('d') or key == 127:  # D or Delete
                img_path = self.images[self.current_idx]
                self.marked_for_deletion.add(img_path)
                print(f"🗑️  标记删除: {img_path.name}")
                self.current_idx += 1
            
            # 撤销标记
            elif key == ord('u'):
                img_path = self.images[self.current_idx]
                if img_path in self.marked_for_deletion:
                    self.marked_for_deletion.remove(img_path)
                    print(f"↩️  取消标记: {img_path.name}")
            
            # 退出
            elif key == ord('q') or key == 27:  # Q or ESC
                break
        
        cv2.destroyAllWindows()
        return self.marked_for_deletion


def main():
    parser = argparse.ArgumentParser(description="交互式标注审核工具")
    parser.add_argument("--visualized", required=True, help="可视化图像目录")
    parser.add_argument("--output", default="marked_for_deletion.txt", 
                       help="输出标记列表文件")
    
    args = parser.parse_args()
    
    visualized_dir = Path(args.visualized)
    if not visualized_dir.exists():
        print(f"❌ 目录不存在: {visualized_dir}")
        return 1
    
    reviewer = AnnotationReviewer(visualized_dir)
    marked_images = reviewer.run()
    
    if marked_images:
        print(f"\n📝 保存标记列表...")
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            for img_path in sorted(marked_images):
                # 写入原始文件名（不含路径）
                f.write(img_path.stem + '\n')
        
        print(f"\n✅ 已保存 {len(marked_images)} 个标记到: {output_file}")
        print(f"\n接下来运行删除脚本:")
        print(f"  python delete_marked_images.py --list {output_file}")
    else:
        print(f"\n✅ 没有标记任何图像")
    
    return 0


if __name__ == "__main__":
    exit(main())
