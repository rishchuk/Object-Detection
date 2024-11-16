from object_detection import ObjectDetection

if __name__ == '__main__':
    video_path = './data/cars.mp4'
    output_path = f'output_{video_path.split("/")[-1]}'
    tracker = ObjectDetection(model_name='yolov5su.pt', video=video_path, output_video=output_path)
    tracker.run()
