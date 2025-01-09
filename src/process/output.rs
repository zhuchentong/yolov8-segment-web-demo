// 输出处理函数
// 将YOLOv8的原始输出转换为可理解的对象数组
// 每个对象包含：边界框坐标、对象类型、置信度和分割掩码
fn process_output(
    outputs: (Array<f32, IxDyn>, Array<f32, IxDyn>),
    img_width: u32,
    img_height: u32,
) -> Vec<(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>)> {
    let (output0, output1) = outputs;
    // 提取边界框和类别信息
    let boxes_output = output0.slice(s![.., 0..84, 0]).to_owned();
    // 处理分割掩码
    let masks_output: Array2<f32> = output1
        .slice(s![.., .., .., 0])
        .to_owned()
        .into_shape((160 * 160, 32))
        .unwrap()
        .permuted_axes([1, 0])
        .to_owned();
    let masks_output2: Array2<f32> = output0.slice(s![.., 84..116, 0]).to_owned();
    let masks = masks_output2
        .dot(&masks_output)
        .into_shape((8400, 160, 160))
        .unwrap()
        .to_owned();

    let mut boxes = Vec::new();
    // 处理每个检测结果
    for (index, row) in boxes_output.axis_iter(Axis(0)).enumerate() {
        let row: Vec<_> = row.iter().map(|x| *x).collect();
        // 获取最高置信度的类别
        let (class_id, prob) = row
            .iter()
            .skip(4)
            .enumerate()
            .map(|(index, value)| (index, *value))
            .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
            .unwrap();

        // 过滤低置信度的检测结果
        if prob < 0.5 {
            continue;
        }

        let mask: Array2<f32> = masks.slice(s![index, .., ..]).to_owned();
        let label = YOLO_CLASSES[class_id];

        // 将边界框坐标转换回原始图像尺寸
        let xc = row[0] / 640.0 * (img_width as f32);
        let yc = row[1] / 640.0 * (img_height as f32);
        let w = row[2] / 640.0 * (img_width as f32);
        let h = row[3] / 640.0 * (img_height as f32);
        let x1 = xc - w / 2.0;
        let x2 = xc + w / 2.0;
        let y1 = yc - h / 2.0;
        let y2 = yc + h / 2.0;

        boxes.push((
            x1,
            y1,
            x2,
            y2,
            label,
            prob,
            process_mask(mask, (x1, y1, x2, y2), img_width, img_height),
        ));
    }

    // 按置信度排序并执行非极大值抑制(NMS)
    boxes.sort_by(|box1, box2| box2.5.total_cmp(&box1.5));
    let mut result = Vec::new();
    while boxes.len() > 0 {
        result.push(boxes[0].clone());
        boxes = boxes
            .iter()
            .filter(|box1| iou(&boxes[0], box1) < 0.7)
            .map(|x| x.clone())
            .collect()
    }
    return result;
}

// 将对象的分割掩码从YOLOv8的160x160输出转换为正确的尺寸
fn process_mask(
    mask: Array2<f32>,
    rect: (f32, f32, f32, f32),
    img_width: u32,
    img_height: u32,
) -> Vec<Vec<u8>> {
    let (x1, y1, x2, y2) = rect;
    // 创建掩码图像
    let mut mask_img = image::DynamicImage::new_rgb8(161, 161);
    let mut index = 0.0;
    // 将掩码值转换为二值图像
    mask.for_each(|item| {
        let color = if *item > 0.0 {
            Rgba::<u8>([255, 255, 255, 1])
        } else {
            Rgba::<u8>([0, 0, 0, 1])
        };
        let y = f32::floor(index / 160.0);
        let x = index - y * 160.0;
        mask_img.put_pixel(x as u32, y as u32, color);
        index += 1.0;
    });
    // 裁剪和调整掩码大小
    mask_img = mask_img.crop(
        (x1 / img_width as f32 * 160.0).round() as u32,
        (y1 / img_height as f32 * 160.0).round() as u32,
        ((x2 - x1) / img_width as f32 * 160.0).round() as u32,
        ((y2 - y1) / img_height as f32 * 160.0).round() as u32,
    );
    mask_img = mask_img.resize_exact((x2 - x1) as u32, (y2 - y1) as u32, FilterType::Nearest);

    // 转换为二维数组格式
    let mut result = vec![];
    for y in 0..(y2 - y1) as usize {
        let mut row = vec![];
        for x in 0..(x2 - x1) as usize {
            let color = mask_img.get_pixel(x as u32, y as u32);
            row.push(*color.0.iter().nth(0).unwrap());
        }
        result.push(row);
    }
    return result;
}

// 计算两个边界框的交并比(IoU)
// IoU用于非极大值抑制过程中判断框的重叠程度
fn iou(
    box1: &(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>),
    box2: &(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>),
) -> f32 {
    return intersection(box1, box2) / union(box1, box2);
}

// 计算两个边界框的并集面积
fn union(
    box1: &(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>),
    box2: &(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>),
) -> f32 {
    let (box1_x1, box1_y1, box1_x2, box1_y2, _, _, _) = *box1;
    let (box2_x1, box2_y1, box2_x2, box2_y2, _, _, _) = *box2;
    let box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
    let box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);
    return box1_area + box2_area - intersection(box1, box2);
}

// 计算两个边界框的交集面积
fn intersection(
    box1: &(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>),
    box2: &(f32, f32, f32, f32, &'static str, f32, Vec<Vec<u8>>),
) -> f32 {
    let (box1_x1, box1_y1, box1_x2, box1_y2, _, _, _) = *box1;
    let (box2_x1, box2_y1, box2_x2, box2_y2, _, _, _) = *box2;
    let x1 = box1_x1.max(box2_x1);
    let y1 = box1_y1.max(box2_y1);
    let x2 = box1_x2.min(box2_x2);
    let y2 = box1_y2.min(box2_y2);
    return (x2 - x1) * (y2 - y1);
}

// YOLOv8可以检测的80个类别标签
const YOLO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];
