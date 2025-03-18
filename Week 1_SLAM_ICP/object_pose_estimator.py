# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# class SimpleObjectPoseEstimator:
#     def __init__(self):
#         # 카메라 매트릭스와 왜곡 계수 (실제 카메라 캘리브레이션 값으로 교체 필요)
#         self.camera_matrix = np.array([
#             [800, 0, 320],
#             [0, 800, 240],
#             [0, 0, 1]
#         ], dtype=np.float32)
        
#         self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
#     def detect_colored_objects(self, image, use_hsv=True):
#         """
#         이미지에서 색상 기반으로 물체 검출
        
#         Args:
#             image: 입력 이미지 (BGR 형식)
#             use_hsv: HSV 색상 공간 사용 여부
            
#         Returns:
#             objects: 검출된 물체 목록 (중심점, 색상, 크기)
#         """
#         # BGR 이미지 복사
#         img_bgr = image.copy()
        
#         # HSV 색상 공간으로 변환 (옵션)
#         if use_hsv:
#             img_color = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
#         else:
#             img_color = img_bgr
            
#         # 결과 저장할 리스트
#         objects = []
        
#         # 여러 색상 범위 정의
#         if use_hsv:
#             # HSV 색상 범위 (색상, 채도, 명도)
#             color_ranges = {
#                 'red1': ([0, 100, 100], [10, 255, 255]),
#                 'red2': ([160, 100, 100], [180, 255, 255]),  # 빨간색은 HSV에서 두 범위로 나뉨
#                 'blue': ([100, 100, 100], [140, 255, 255]),
#                 'green': ([40, 100, 100], [80, 255, 255]),
#                 'yellow': ([20, 100, 100], [40, 255, 255]),
#                 'orange': ([10, 100, 100], [20, 255, 255])
#             }
#         else:
#             # BGR 색상 범위
#             color_ranges = {
#                 'red': ([0, 0, 100], [50, 50, 255]),
#                 'blue': ([100, 50, 0], [255, 100, 50]),
#                 'green': ([0, 100, 0], [50, 255, 50]),
#                 'yellow': ([0, 200, 200], [50, 255, 255]),
#                 'orange': ([0, 100, 200], [50, 140, 255])
#             }
        
#         # 각 색상 범위에 대해 물체 검출
#         for color_name, (lower, upper) in color_ranges.items():
#             # 색상 범위에 따라 마스크 생성
#             mask = cv2.inRange(img_color, np.array(lower), np.array(upper))
            
#             # 노이즈 제거
#             kernel = np.ones((5, 5), np.uint8)
#             mask = cv2.erode(mask, kernel, iterations=1)
#             mask = cv2.dilate(mask, kernel, iterations=2)
            
#             # 윤곽선 찾기
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
#             # 각 윤곽선 처리
#             for contour in contours:
#                 # 면적 계산
#                 area = cv2.contourArea(contour)
                
#                 # 작은 영역 무시
#                 if area < 500:
#                     continue
                    
#                 # 중심점 계산
#                 M = cv2.moments(contour)
#                 if M["m00"] != 0:
#                     cx = int(M["m10"] / M["m00"])
#                     cy = int(M["m01"] / M["m00"])
#                 else:
#                     cx, cy = 0, 0
                    
#                 # 경계 사각형 찾기
#                 x, y, w, h = cv2.boundingRect(contour)
                
#                 # 물체 정보 저장
#                 objects.append({
#                     'center': (cx, cy),
#                     'color': color_name,
#                     'area': area,
#                     'bbox': (x, y, w, h),
#                     'contour': contour
#                 })
        
#         return objects
    
#     def estimate_pose(self, objects, image):
#         """
#         검출된 물체의 포즈를 추정합니다.
        
#         Args:
#             objects: 검출된 물체 목록
#             image: 원본 이미지
            
#         Returns:
#             image_with_poses: 포즈 시각화된 이미지
#         """
#         result_image = image.copy()
        
#         # 각 물체에 대해 포즈 추정
#         for obj in objects:
#             # 물체의 중심점
#             center = obj['center']
            
#             # 물체의 색상에 따라 표시 색상 지정
#             if obj['color'] == 'red' or obj['color'] == 'red1' or obj['color'] == 'red2':
#                 color = (0, 0, 255)
#             elif obj['color'] == 'blue':
#                 color = (255, 0, 0)
#             elif obj['color'] == 'green':
#                 color = (0, 255, 0)
#             elif obj['color'] == 'yellow':
#                 color = (0, 255, 255)
#             elif obj['color'] == 'orange':
#                 color = (0, 165, 255)
#             else:
#                 color = (255, 255, 255)
            
#             # 윤곽선 그리기
#             cv2.drawContours(result_image, [obj['contour']], 0, color, 2)
            
#             # 중심점 표시
#             cv2.circle(result_image, center, 5, color, -1)
            
#             # 색상 및 좌표 텍스트 표시
#             text = f"{obj['color']}: ({center[0]}, {center[1]})"
#             cv2.putText(result_image, text, (center[0] - 50, center[1] - 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
#             # 간단한 3D 좌표 추정 (Z는 물체 크기에 반비례한다고 가정)
#             # 실제로는 더 복잡한 계산이 필요합니다
#             z_estimate = 10000 / obj['area']
            
#             # 3D 좌표 표시
#             x_3d = center[0] - 320  # 이미지 중심으로부터의 X 편차
#             y_3d = center[1] - 240  # 이미지 중심으로부터의 Y 편차
            
#             text_3d = f"3D: ({x_3d:.1f}, {y_3d:.1f}, {z_estimate:.1f})"
#             cv2.putText(result_image, text_3d, (center[0] - 50, center[1] + 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
#         return result_image

# def main():
#     # 이미지 파일 경로 (여러분이 찍은 테이블 위 물체 사진으로 교체)
#     image_path = "colorful_objects.jpg"
    
#     # 이미지 읽기
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"이미지를 읽을 수 없습니다: {image_path}")
#         return
    
#     # 이미지 크기 조정 (필요시)
#     image = cv2.resize(image, (640, 480))
    
#     # 물체 포즈 추정기 초기화
#     estimator = SimpleObjectPoseEstimator()
    
#     # BGR 색상 공간 사용
#     objects_bgr = estimator.detect_colored_objects(image, use_hsv=False)
#     result_bgr = estimator.estimate_pose(objects_bgr, image)
    
#     # HSV 색상 공간 사용
#     objects_hsv = estimator.detect_colored_objects(image, use_hsv=True)
#     result_hsv = estimator.estimate_pose(objects_hsv, image)
    
#     # 결과 표시
#     plt.figure(figsize=(15, 10))
    
#     plt.subplot(1, 3, 1)
#     plt.title("원본 이미지")
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
#     plt.subplot(1, 3, 2)
#     plt.title(f"BGR 색상 공간 (검출: {len(objects_bgr)}개)")
#     plt.imshow(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
    
#     plt.subplot(1, 3, 3)
#     plt.title(f"HSV 색상 공간 (검출: {len(objects_hsv)}개)")
#     plt.imshow(cv2.cvtColor(result_hsv, cv2.COLOR_BGR2RGB))
    
#     plt.tight_layout()
#     plt.savefig("pose_estimation_results.png")
#     plt.show()
    
#     # 검출된 물체 정보 출력
#     print("BGR 색상 공간으로 검출된 물체:")
#     for i, obj in enumerate(objects_bgr):
#         print(f"물체 {i+1}: {obj['color']}, 중심점: {obj['center']}, 면적: {obj['area']}")
    
#     print("\nHSV 색상 공간으로 검출된 물체:")
#     for i, obj in enumerate(objects_hsv):
#         print(f"물체 {i+1}: {obj['color']}, 중심점: {obj['center']}, 면적: {obj['area']}")
    
#     # 결과 이미지 저장
#     cv2.imwrite("result_bgr.jpg", result_bgr)
#     cv2.imwrite("result_hsv.jpg", result_hsv)
    
#     print("\nBGR vs HSV 비교:")
#     print(f"BGR 검출 수: {len(objects_bgr)}")
#     print(f"HSV 검출 수: {len(objects_hsv)}")
    
#     # 더 나은 컬러 모델 제안
#     if len(objects_hsv) > len(objects_bgr):
#         print("분석 결과: HSV 색상 공간이 더 많은 물체를 검출했습니다.")
#     else:
#         print("분석 결과: BGR 색상 공간이 더 많거나 같은 수의 물체를 검출했습니다.")

# if __name__ == "__main__":
#     main()