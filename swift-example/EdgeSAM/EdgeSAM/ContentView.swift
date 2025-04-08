//
//  ContentView.swift
//  EdgeSAM
//
//  Created by seung on 2025/04/02.
//

import SwiftUI
import CoreML

struct ContentView: View {
    @State var images: [UIImage]?
    var body: some View {
        VStack {
            Image(uiImage: UIImage(named: "truck")!)
                .resizable()
                .renderingMode(.original)
                .aspectRatio(contentMode: .fit)
            Text("------------------------------------")
            if images != nil {
                ForEach(images!, id: \.self) { image in
                    ZStack {
                        Image(uiImage: UIImage(named: "truck")!)
                            .resizable()
                            .frame(width: 200,height: 200)

                        Image(uiImage: image)
                            .resizable()
                            .opacity(0.5)
                            .frame(width: 200,height: 200)
                    }
                }
            }
        }
        .padding()
        .onAppear() {
            images = getResult()
        }
    }
}

#Preview {
    ContentView()
}

func getResult() -> [UIImage]? {

    let inferenceSize:CGSize = CGSize(
        width: 1024,
        height: 1024
    )
    
    let uiImage:UIImage = UIImage(named: "truck")!
    let originalImage:CGImage! = uiImage.cgImage
    let originalImageSize = CGSize(
        width: originalImage.width,
        height: originalImage.height
    )
    
    ExecutionMeasurer.start(.paddingAndResizing)
    let vImageWrapper = VImageWrapper(cgImage: originalImage)
    let cgImage = vImageWrapper.resizeStretch(
        cropRect: CGRect(x: 0, y: 0, width: originalImageSize.width, height: originalImageSize.height),
        desiredSize: inferenceSize
    )!
    
//    return UIImage(cgImage: cgImage)
//    let cgImage = vImageWrapper.paddingAndResizing(
//        to: inferenceSize,
//        targetRectInSource: CGRect(x: 0, y: 0, width: originalImageSize.width, height: originalImageSize.height),
//        degreeToRotateInClockwise: 0
//    )!
    ExecutionMeasurer.stop(.paddingAndResizing)
    
    ExecutionMeasurer.start(.conversionRGB)
    //inputArray-RGBRGB  0.196078431372549 0.2 0.2156862745098039
    // inputArray-RRGGBB 0.196078431372549 0.2980392156862745 0.5607843137254902

    let inputArray = convertToRGB(cgImage: cgImage,
                                  inputShape:[1,3,
                                              inferenceSize.width as NSNumber,
                                              inferenceSize.height as NSNumber],
                                  rgbFormat: .RGBRGB)!
    ExecutionMeasurer.start(.conversionRGB)
    
    // inputArray 0.2431373 0.2392157 0.5490196
    guard let inputArray2 = prepareImageInputMLMultiArray(originalImage:  uiImage) else { return  nil }
    
    print("inputArray", inputArray)
    print("inputArray.shape", inputArray.shape)
//    inputArray 0.2431373 0.2392157 0.5490196
    
    print("inputArray", inputArray[0], inputArray[1], inputArray[2], inputArray[3])
    var offset = 1024 * 1024
    print("inputArray offset", inputArray[0], inputArray[offset * 1], inputArray[offset * 2])
    
    print("inputArray", inputArray[offset+0], inputArray[offset+1], inputArray[offset+2], inputArray[offset+3])
    print("inputArray offset", inputArray[0 + 500], inputArray[offset * 1 + 500], inputArray[offset * 2 + 500])
    
//    fatalError()
    let allZeroInput = try! MLMultiArray(shape: [1,3,1024,1024], dataType: .float32)
    let allOneInput = try! MLMultiArray(shape: [1,3,1024,1024], dataType: .float32)
    for i in 0..<allOneInput.count {
        allOneInput[i] = 200
    }
    for input in [allZeroInput, allOneInput, inputArray, inputArray2] {
        let encoderInput = edge_sam_3x_encoderInput(image: input)
        
        let encoder = try! edge_sam_3x_encoder()
        let options = MLPredictionOptions()
        
//        options.outputBackings
        let encoderOutput = try! encoder.prediction(input: encoderInput, options: options)
        let imageEmbeddings = encoderOutput.image_embeddings
        
        //    guard let imageEmbeddings = encode(imageArray: inputArray) else { return nil }
        
        print("---------------------------------------")
        print("imageEmbeddings", imageEmbeddings)
        print("imageEmbeddings", imageEmbeddings[0], imageEmbeddings[1], imageEmbeddings[2], imageEmbeddings[3])
        
        offset = 64 * 64
        print("imageEmbeddings offset", imageEmbeddings[0], imageEmbeddings[offset * 1], imageEmbeddings[offset * 2])
        print("---------------------------------------")
    }
//    fatalError()
    let encoderInput = edge_sam_3x_encoderInput(image: inputArray)
    
    let encoder = try! edge_sam_3x_encoder()
    let options = MLPredictionOptions()
    let encoderOutput = try! encoder.prediction(input: encoderInput, options: options)
    
    
//    imageEmbeddings 0.3017645 0.2324331 0.2324331 0.2324331
//    imageEmbeddings offset 0.3017645 0.1857381 0.09450045
    
//    imageEmbeddings 0.3017645 0.2324331 0.2324331 0.2324331
//    imageEmbeddings offset 0.3017645 0.1857381 0.09450045
    
//    imageEmbeddings 0.3017645 0.2324331 0.2324331 0.2324331
//    imageEmbeddings offset 0.3017645 0.1857381 0.09450045
    let imageEmbeddings = encoderOutput.image_embeddings
    
//    guard let imageEmbeddings = encode(imageArray: inputArray) else { return nil }
    
    print("imageEmbeddings", imageEmbeddings)
    print("imageEmbeddings", imageEmbeddings[0], imageEmbeddings[1], imageEmbeddings[2], imageEmbeddings[3])
    
    offset = 64 * 64
    print("imageEmbeddings offset", imageEmbeddings[0], imageEmbeddings[offset * 1], imageEmbeddings[offset * 2])
    
//    fatalError()
    
    let inputPoints:[[Float]] = [[
        500 / Float(originalImageSize.width) * 1024,
        675 / Float(originalImageSize.height) * 1024
    ]]
//    let inputPoints:[[Float]] = [[500, 375]]
    let inputLabels:[Int] = [1]
    /// image_embeddings as 1 × 256 × 64 × 64 4-dimensional array of floats
    /// point_coords as 1 × 1 × 2 3-dimensional array of floats
    /// point_labels as 1 by 1 matrix of floats
    guard
        let pointCoords = convertToMLPointCoords(pointCoords: inputPoints),
        let pointLabels = convertToMLPointLabels(pointLabels: inputLabels)
    else {
        print("Failed to convert input data to MLMultiArray")
        return nil
    }
    print("pointCoords", pointCoords[0], pointCoords[1])
    print("pointCoords.shape", pointCoords.shape)
    print("pointLabels.shape", pointLabels.shape)
    
    guard let masks = decode(
        imageEmbedding: imageEmbeddings,
        pointCoords: pointCoords,
        pointLabels: pointLabels
    ) else {
        print("Failed to decode")
        return nil
    }
    print("masks", masks)
    print("masks.shape", masks.shape)
    var i = 0
    func extractBoolMask(from multiArray: MLMultiArray, channel: Int) -> [[Bool]]? {
        let shape = multiArray.shape.map { $0.intValue }
        
        guard shape.count == 4,
              shape[0] == 1,
              channel < shape[1] else { return nil }
        
        let height = shape[2]
        let width = shape[3]
        if i == 0 {
            i += 1
            print("width, height", width, height)
        }
        
        var result = Array(repeating: Array(repeating: false, count: width), count: height)
        
        let pointer = UnsafeMutablePointer<Float32>(OpaquePointer(multiArray.dataPointer))
        
        let channelOffset = channel * height * width
        
        for y in 0..<height {
            for x in 0..<width {
                let flatIndex = channelOffset + y * width + x
                result[y][x] = pointer[flatIndex] >= 0
            }
        }
        
        return result
    }
    var images:[UIImage] = []
    for i in 0..<4 {
//        let mask = masks[i]
        
        
        if let image = showMask(
            mask: extractBoolMask(from: masks, channel: i)!
        ) {
            print("done!!!!!!!!2222")
//            return [image]
            images.append(image)
        }
        
    }
    return images
//    return nil
    if let image = showMask(
        mask: extractBoolMask(from: masks, channel: 0)!
    ) {
        print("done!!!!!!!!")
        return [image]
    }
    
    return nil
    
    
    guard let slice = extractMaskSlice(from: masks, index: 0),
          let maskImage = postprocessMask(slice, targetSize: originalImageSize) else {
        print("Failed to postprocess mask")
        return nil
    }
    print("slice.shape", slice.shape)
    
    printMLMultiArrayFlat(slice)
    func printMLMultiArrayFlat(_ array: MLMultiArray) {
        for i in 0..<array.count {
            print("\(i) float: \(array[i].floatValue)")
//            print("\(i): \(array[i].doubleValue)")
            print("\(i) bool: \(array[i].boolValue)")
        }
    }
    
//    guard let maskImage = maskImageWithBitImage(originalImage: originalImage, maskVectors: convert(mlMultiArray: slice)) else { return nil }
//
//    return maskImage
    
//    if let floatMask = extractMaskChannel(masks, channel: 1) {
//        let gray = convertToUInt8Grayscale(floatMask)
//        if let maskImage = makeGrayscaleImage(from: gray) {
//            let resized = resizeImage(maskImage, to: CGSize(width: 1024, height: 1024))
//            return resized
//        }
//    }
    
    return nil
    
    func showMask(mask: [[Bool]], randomColor: Bool = false) -> UIImage? {
        let height = mask.count
        guard height > 0 else { return nil }
        let width = mask[0].count
        
        // Generate color
        let color: (r: CGFloat, g: CGFloat, b: CGFloat, a: CGFloat)
        if randomColor {
            color = (
                r: CGFloat.random(in: 0...1),
                g: CGFloat.random(in: 0...1),
                b: CGFloat.random(in: 0...1),
                a: 0.6
            )
        } else {
            color = (
                r: 30/255,
                g: 144/255,
                b: 255/255,
                a: 0.6
            )
        }
        
        // Create pixel data
        let bytesPerPixel = 4
        let imageData = UnsafeMutablePointer<UInt8>.allocate(capacity: width * height * bytesPerPixel)
        defer { imageData.deallocate() }
        
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * bytesPerPixel
                if mask[y][x] {
                    imageData[offset + 0] = UInt8(color.r * 255)
                    imageData[offset + 1] = UInt8(color.g * 255)
                    imageData[offset + 2] = UInt8(color.b * 255)
                    imageData[offset + 3] = UInt8(color.a * 255)
                } else {
                    imageData[offset + 0] = 0
                    imageData[offset + 1] = 0
                    imageData[offset + 2] = 0
                    imageData[offset + 3] = 0
                }
            }
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: imageData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * bytesPerPixel,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        
        guard let cgImage = context?.makeImage() else { return nil }
        return UIImage(cgImage: cgImage)
    }
    func imageFromBooleanMask(_ mask: MLMultiArray, channel: Int) -> UIImage? {
        // Validate shape
        let batch = mask.shape[0].intValue
        let channels = mask.shape[1].intValue
        let height = mask.shape[2].intValue
        let width = mask.shape[3].intValue
        
        guard mask.shape.count == 4, batch == 1, channel < channels else {
            print("falllllsss")
            return nil
        }
        
        let pixels = UnsafeMutablePointer<UInt8>.allocate(capacity: width * height)
        defer { pixels.deallocate() }
        
        // Flatten MLMultiArray to access elements
        let ptr = UnsafeMutablePointer<Bool>(OpaquePointer(mask.dataPointer))
        
        // Calculate offset for the selected channel
        let channelOffset = channel * height * width
        
        for y in 0..<height {
            for x in 0..<width {
                let idx = channelOffset + y * width + x
                pixels[y * width + x] = ptr[idx] ? 255 : 0
            }
        }
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let context = CGContext(data: pixels,
                                width: width,
                                height: height,
                                bitsPerComponent: 8,
                                bytesPerRow: width,
                                space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.none.rawValue)
        
        guard let cgImage = context?.makeImage() else { return nil }
        return UIImage(cgImage: cgImage)
    }
    
    func extractMaskChannel(_ multiArray: MLMultiArray, channel: Int) -> [Float]? {
        let channels = multiArray.shape[1].intValue
        let height = multiArray.shape[2].intValue
        let width = multiArray.shape[3].intValue
        
        guard channel < channels else { return nil }
        
        var result = [Float](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let idx = channel * height * width + y * width + x
                let value = multiArray[idx].floatValue
                result[y * width + x] = value
            }
        }
        return result
    }
    func convertToUInt8Grayscale(_ floats: [Float]) -> [UInt8] {
        return floats.map { val in
            if val.isNaN || !val.isFinite { return 0 }
            let clamped = min(max(val, 0.0), 1.0)
            return UInt8(clamped * 255.0)
        }
    }
    func makeGrayscaleImage(from pixelBuffer: [UInt8], width: Int = 256, height: Int = 256) -> UIImage? {
        let bytesPerRow = width
        let colorSpace = CGColorSpaceCreateDeviceGray()
        
        guard let provider = CGDataProvider(data: Data(pixelBuffer) as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
              ) else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
    func resizeImage(_ image: UIImage, to targetSize: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 0.0)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized
    }
    
    func convert(mlMultiArray: MLMultiArray) -> [UInt8] {
        let count = mlMultiArray.count
        let pointer = UnsafeMutablePointer<UInt8>(OpaquePointer(mlMultiArray.dataPointer))
        let buffer = UnsafeBufferPointer(start: pointer, count: count)
        return Array(buffer)
    }
    

    func maskImageWithBitImage(originalImage:CGImage, maskVectors:[UInt8]) -> UIImage? {
        let maskData = Data(maskVectors)
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        guard let providerRef = CGDataProvider(data: maskData as CFData) else { return nil }
        //        let width:Int = 256,
        //            height:Int = 256
        //        let context = CGContext(data: nil,
        //                                width: width,
        //                                height: height,
        //                                bitsPerComponent: 8,
        //                                bytesPerRow: width,
        //                                space: CGColorSpaceCreateDeviceGray(),
        //                                bitmapInfo: bitmapInfo.rawValue)!
        //        for y in 0..<height {
        //            for x in 0..<width {
        //                let pixelValue = maskVectors[height * y + x]
        //                context.setFillColor(CGColor(gray: CGFloat(pixelValue) / 255.0, alpha: 1.0))
        //                context.fill(CGRect(x: x, y: height-y, width: 1, height: 1))
        //            }
        //        }
        //        let maskImage = context.makeImage()!
        //        return UIImage(cgImage: maskImage)
        
        guard let maskImage = CGImage(width: Int(256),
                                      height: Int(256),
                                      bitsPerComponent: 8,
                                      bitsPerPixel: 8,
                                      bytesPerRow: Int(256),
                                      space: CGColorSpaceCreateDeviceGray(),
                                      bitmapInfo: bitmapInfo,
                                      provider: providerRef,
                                      decode: nil,
                                      shouldInterpolate: false,
                                      intent: .defaultIntent) else { return nil }
//        guard let maskedImage = originalImage.masking(maskImage) else { return nil }
        return UIImage(cgImage: maskImage)
    }
    
    func extractMaskSlice(from array: MLMultiArray, index: Int) -> MLMultiArray? {
        let height = 256
        let width = 256
        guard index >= 0 && index < 4 else { return nil }
        
        // Allocate new MLMultiArray [1, 1, 256, 256] with only one slice
        guard let singleMask = try? MLMultiArray(shape: [1, 1, 256, 256], dataType: .float32) else {
            return nil
        }
        
        for y in 0..<height {
            for x in 0..<width {
                let originalIndex = index * height * width + y * width + x
                let newIndex = y * width + x
                singleMask[newIndex] = array[originalIndex]
            }
        }
        
        return singleMask
    }

    func postprocessMask(_ maskArray: MLMultiArray, targetSize: CGSize) -> UIImage? {
        let shape = maskArray.shape
        guard shape.count == 4,
              shape[0].intValue == 1,
              shape[1].intValue == 1,
              shape[2].intValue == 256,
              shape[3].intValue == 256 else {
            print("Unexpected mask shape: \(shape)")
            return nil
        }
        
        let height = 256
        let width = 256
        
        // Flattened MLMultiArray → grayscale UInt8 pixels
        var pixelBuffer = [UInt8](repeating: 0, count: width * height)
        
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                let value = maskArray[index].floatValue
                
                // Handle NaN or inf
                if value.isNaN || !value.isFinite {
                    pixelBuffer[index] = 0
                    continue
                }
                
                let clamped = min(max(value, 0.0), 1.0)
                pixelBuffer[index] = UInt8(clamped * 255.0)
            }
        }

        
        // Create CGImage from grayscale pixel buffer
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let dataProvider = CGDataProvider(data: Data(pixelBuffer) as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 8,
                bytesPerRow: width,
                space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: dataProvider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent) else {
            return nil
        }
        
        let baseImage = UIImage(cgImage: cgImage)
        
        // Resize to target size
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 0.0)
        baseImage.draw(in: CGRect(origin: .zero, size: targetSize))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return resizedImage
    }
    
    func convertToMLPointCoords(pointCoords: [[Float]]) -> MLMultiArray? {
        let N = pointCoords.count
        
        guard let array = try? MLMultiArray(shape: [1, NSNumber(value: N), 2], dataType: .float32) else {
            return nil
        }
        for (i, coord) in pointCoords.enumerated() {
            let baseIndex = i * 2
            array[baseIndex + 0] = NSNumber(value: coord[0])
            array[baseIndex + 1] = NSNumber(value: coord[1])
        }
        return array
    }
    func convertToMLPointLabels(pointLabels: [Int]) -> MLMultiArray? {
        let N = pointLabels.count
        
        guard let array = try? MLMultiArray(shape: [1, NSNumber(value: N)], dataType: .int32) else {
            return nil
        }
        for (i, label) in pointLabels.enumerated() {
            array[i] = NSNumber(value: label)
        }
        return array
    }
    
    
    /// image as 1 × 3 × 1024 × 1024 4-dimensional array of floats
    func encode(imageArray: MLMultiArray) -> MLMultiArray? {
        let encoderInput = edge_sam_3x_encoderInput(image: imageArray)

        do {
            let encoder = try edge_sam_3x_encoder()
            let options = MLPredictionOptions()
            let encoderOutput = try encoder.prediction(input: encoderInput, options: MLPredictionOptions())
            
            return encoderOutput.image_embeddings
        } catch {
            print("Failed to perform EdgeSAM encoder inference", error.localizedDescription)
            return nil
        }
    }
    
    func decode(
        imageEmbedding: MLMultiArray,
        pointCoords: MLMultiArray,
        pointLabels: MLMultiArray
    ) -> MLMultiArray? {
        
        print("imageEmbedding.shape", imageEmbedding.shape)
        
        print("Fisrt imageEmbedding:", imageEmbedding[0])
        print("First coord:", pointCoords[0], pointCoords[1])
        print("First label:", pointLabels[0])
        
  
        
        do {
            let decoderInput = edge_sam_3x_decoderInput(
                image_embeddings: imageEmbedding,
                point_coords: pointCoords,
                point_labels: pointLabels
            )
            
            let decoder = try edge_sam_3x_decoder()
            let decoderOutput =  try decoder.prediction(input: decoderInput)
            printMLMultiArrayFlat(decoderOutput.scores)
//            0: nan
//            1: nan
//            2: nan
//            3: nan
            print("decoderOutput.masks[0] floatValue", decoderOutput.masks[0].floatValue)
            print("decoderOutput.masks[0] boolValue", decoderOutput.masks[0].boolValue)
            print(decoderOutput.masks.shape)
//            fatalError()
            return decoderOutput.masks
        } catch {
            return nil
        }
        
        
        return nil
        
        
    }
}
