//
//  VImageWrapper.swift
//  EdgeSAM
//
//  Created by seung on 2025/04/03.
//

import Foundation
import Accelerate
import CoreML

//#uiImage: UIImageの画像
//let imagePixel = uiImage.getPixelRgb()
// https://www.ralfebert.com/ios/examples/image-processing/uiimage-raw-pixels/
extension CGImage {
    //pixelBUfferに変換(RGB値)
    func getPixelRgb(rgbFormat:RGBFormat) -> [Double] {
        let cgImage = self
        let bytesPerRow = cgImage.bytesPerRow
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let pixelData = cgImage.dataProvider!.data! as Data
        
        var res:[Double] = Array<Double>(repeating:0,count:width*height*3)
        
        for j in 0..<height {
            for i in 0..<width {
                let pixelInfo = bytesPerRow * j + i * bytesPerPixel
                let r = Double(pixelData[pixelInfo])
                let g = Double(pixelData[pixelInfo+1])
                let b = Double(pixelData[pixelInfo+2])
                
                switch rgbFormat {
                case .RRGGBB:
                    let idx = i + j*width
                    res[idx] = r / 255
                    res[width*height+idx] = g / 255
                    res[width*height*2+idx] = b / 255
                case .RGBRGB:
                    let idx = 3*width*j + i*3
                    res[idx] = r / 255
                    res[idx+1] = g / 255
                    res[idx+2] = b / 255
                case .BGRBGR:
                    let idx = 3*width*j + i*3
                    res[idx] = b / 255
                    res[idx+1] = g / 255
                    res[idx+2] = r / 255
                }
            }
        }
        return res
    }
}

enum RGBFormat {
    case RRGGBB
    case RGBRGB
    case BGRBGR
}
func convertToRGB(cgImage:CGImage, inputShape:[NSNumber], rgbFormat:RGBFormat) -> MLMultiArray? {
    let imagePixel = cgImage.getPixelRgb(rgbFormat: rgbFormat)
    return imagePixel.withUnsafeBufferPointer { buffer -> MLMultiArray? in
        guard let imagePointer = buffer.baseAddress else { return nil }
        let mlArray = try! MLMultiArray(shape: inputShape, dataType: MLMultiArrayDataType.double)
        mlArray.dataPointer.initializeMemory(as: Double.self, from: imagePointer, count: imagePixel.count)
        return mlArray
    }
}


func crop(buffer: vImage_Buffer, to region: CGRect) -> vImage_Buffer? {
    guard region.minX >= 0
            && region.minY >= 0
            && region.maxX <= CGFloat(buffer.width)
            && region.maxY <= CGFloat(buffer.height)
    else {
        print("region", region.minX, region.minY, region.maxX, region.maxY)
        print("invalid size is given", region, buffer.width, buffer.height)
        return nil
    }
    
    if region.minX == 0 && region.maxX == region.width
        && region.minY == 0 && region.maxY == region.height {
        print("No change is required for cropping")
        return buffer
    }
    
    var croppedBuffer = vImage_Buffer()
    
    let startRow = Int(region.minY)
    let startCol = Int(region.minX)
    let height = Int(region.height)
    let width = Int(region.width)
    
    /*
     - startCol * 4
     In many image processing libraries, each pixel is represented by multiple bytes, where each byte stores a different color channel or additional information (such as alpha channel for transparency). The factor of 4 suggests that the buffer is organized in a format where each pixel occupies 4 bytes, likely representing red, green, blue, and alpha channels.
     */
    let start = buffer.rowBytes * startRow + startCol * 4
    croppedBuffer.data = buffer.data.advanced(by: start)
    croppedBuffer.height = vImagePixelCount(height)
    croppedBuffer.width = vImagePixelCount(width)
    croppedBuffer.rowBytes = buffer.rowBytes
    return croppedBuffer
}


// https://www.kodeco.com/19456196-swift-accelerate-and-vimage-getting-started
struct VImageWrapper {
    let cgImage:CGImage
    
    init(cgImage:CGImage) {
        self.cgImage = cgImage
    }
    
    /**
     - usages
     guard let (sourceBuffer, sourceFormat) = getSourceBufferAndFormat() else { return nil }
     defer { sourceBuffer.free() }
     */
    fileprivate func getSourceBufferAndFormat() -> (vImage_Buffer, vImage_CGImageFormat)? {
        guard let colorSpace = cgImage.colorSpace else { return nil }
        var format = vImage_CGImageFormat(bitsPerComponent: numericCast(cgImage.bitsPerComponent),
                                          bitsPerPixel: numericCast(cgImage.bitsPerPixel),
                                          colorSpace: Unmanaged.passUnretained(colorSpace),
                                          bitmapInfo: cgImage.bitmapInfo,
                                          version: 0,
                                          decode: nil,
                                          renderingIntent: .absoluteColorimetric)
        var sourceBuffer = vImage_Buffer()
        let error = vImageBuffer_InitWithCGImage(&sourceBuffer, &format, nil, cgImage, numericCast(kvImageNoFlags))
        guard error == kvImageNoError else {
            sourceBuffer.free()
            print("Failed to convert to vImageBuffer")
            return nil
        }
        return (sourceBuffer, format)
    }
}

extension VImageWrapper {
    /**
     Crop first and then rotate
     */
    func paddingAndResizing(to size:CGSize,
                            targetRectInSource:CGRect,
                            degreeToRotateInClockwise angleInDegree:Double) -> CGImage? {
        guard let (sourceBuffer, sourceFormat) = getSourceBufferAndFormat() else { return nil }
        defer { sourceBuffer.free() }
        guard let vImage = crop(buffer: sourceBuffer, to: targetRectInSource) else {
            print("Failed to get vImage")
            return nil
        }
        //        defer { vImage.free() }
        let destinationWidth = Int(size.width),
            destinationHeight = Int(size.height)
        
        guard
            var destinationBuffer = try? vImage_Buffer(width: destinationWidth,
                                                       height: destinationHeight,
                                                       bitsPerPixel: 32)
        else { return nil }
        defer { destinationBuffer.free() }
        let backgroundColor: [Pixel_8] = [114, 114, 114, 255] // r g b a
        
        resizeAndPaddingBuffer(source: vImage,
                               destination: &destinationBuffer,
                               angleInDegrees: -angleInDegree,
                               backgroundColor: backgroundColor)
        
        guard let outputCGImage = try? destinationBuffer.createCGImage(format: sourceFormat ) else {
            print("Error: Failed to create CGImage from vImage_Buffer.")
            return nil
        }
        
        return outputCGImage
    }
    
    /// angleInDegree: Counterclock-wise
    private func resizeAndPaddingBuffer(source: vImage_Buffer,
                                        destination: inout vImage_Buffer,
                                        angleInDegrees: Double,
                                        backgroundColor: [Pixel_8] = [0, 127, 127, 127]) {
        // 1. Convert the specified angle in degrees to radians.
        let angle = Measurement(value: angleInDegrees,
                                unit: UnitAngle.degrees)
        let radians = CGFloat(angle.converted(to: .radians).value)
        
        // 2. Calculate the scale based on the bounding box of the rotated image.
        
        let rotatedBoundingBox = CGRect(origin: .zero, size: source.size)
            .applying(CGAffineTransform(rotationAngle: radians))
        let scale = min(destination.size.width / rotatedBoundingBox.size.width,
                        destination.size.height / rotatedBoundingBox.size.height)
        
        // MARK: target rect
        var ratio_w:Double = 1,
            ratio_h:Double = 1
        let source_W = rotatedBoundingBox.size.width,
            source_H = rotatedBoundingBox.size.height
        if source_W > source_H {
            ratio_h = source_H / source_W
        } else {
            ratio_w = source_W / source_H
        }
        let cgTransform = CGAffineTransform.identity
            .translatedBy(x: destination.size.width/2 * ratio_w,
                          y: destination.size.height/2 * ratio_h)
            .rotated(by: radians)
            .scaledBy(x: scale, y: scale)
            .translatedBy(x: -source.size.width/2,
                          y: -source.size.height/2)
        
        var vImageTransform = vImage_CGAffineTransform(
            a: Double(cgTransform.a),
            b: Double(cgTransform.b),
            c: Double(cgTransform.c),
            d: Double(cgTransform.d),
            tx: Double(cgTransform.tx),
            ty: Double(cgTransform.ty))
        
        // 3. Apply the transform to `source` and write the result to `destination`.
        let error = withUnsafePointer(to: source) { srcPointer in
            vImageAffineWarpCG_ARGB8888(srcPointer,
                                        &destination,
                                        nil,
                                        &vImageTransform,
                                        backgroundColor,
                                        vImage_Flags(kvImageBackgroundColorFill))
        }
        guard error == kvImageNoError else { fatalError("Failed to resize and add padding") }
    }
    
    func resizeStretch(cropRect requestedCropRect:CGRect, desiredSize:CGSize) -> CGImage? {
        guard let (sourceBuffer, sourceFormat) = getSourceBufferAndFormat() else { return nil }
        defer { sourceBuffer.free() }
        
        //        if requestedCropRect.size.width + requestedCropRect.minX <= CGFloat(cgImage.width)
        //            && requestedCropRect.minX >= 0
        //            && requestedCropRect.size.height + requestedCropRect.minY <= CGFloat(cgImage.height)
        //            && requestedCropRect.minY >= 0 {
        //            print("[DEBUG] -- cropRect.minX", requestedCropRect.minX)
        //            print("[DEBUG] -- cropRect.size.width", requestedCropRect.size.width)
        //            print("[DEBUG] -- cropRect.size.width + cropRect.minX", requestedCropRect.size.width + requestedCropRect.minX)
        //            print("[DEBUG] -- cgImage.width", cgImage.width)
        //
        //            print("[DEBUG] -- cropRect.minY", requestedCropRect.minY)
        //            print("[DEBUG] -- cropRect.size.height", requestedCropRect.size.height)
        //            print("[DEBUG] -- cropRect.size.height + cropRect.minY", requestedCropRect.size.height + requestedCropRect.minY)
        //            print("[DEBUG] -- cgImage.height", cgImage.height)
        //        }
        
        let minX:CGFloat = max(0,requestedCropRect.minX),
            minY:CGFloat = max(0,requestedCropRect.minY),
            maxX:CGFloat = min(requestedCropRect.minX + requestedCropRect.size.width, CGFloat(cgImage.width)),
            maxY:CGFloat = min(requestedCropRect.minY + requestedCropRect.size.height, CGFloat(cgImage.height))
        
        let cropRect = CGRect(x: minX,
                              y: minY,
                              width: maxX-minX,
                              height: maxY-minY)
        assert(
            cropRect.size.width + cropRect.minX <= CGFloat(cgImage.width)
            && cropRect.minX >= 0
            && cropRect.size.height + cropRect.minY <= CGFloat(cgImage.height)
            && cropRect.minY >= 0
        )
        let W = Double(cgImage.width),
            H = Double(cgImage.height)
        let WW = desiredSize.width,
            HH = desiredSize.height
        
        let scaleX = WW / cropRect.size.width,
            scaleY = HH / cropRect.size.height
        let cropRectLeft = cropRect.minX
        let cropRectTop = H - cropRect.maxY
        
        let cgTransform = CGAffineTransform.identity
            .translatedBy(x: -cropRectLeft * scaleX,
                          y: -cropRectTop * scaleY)
            .scaledBy(x: scaleX, y: scaleY)
        
        
        guard var destinationBuffer = try? vImage_Buffer(
            width: Int(WW),
            height: Int(HH),
            bitsPerPixel: 32) else { return nil }
        defer { destinationBuffer.free() }
        
        var vImageTransform = vImage_CGAffineTransform(
            a: Double(cgTransform.a),
            b: Double(cgTransform.b),
            c: Double(cgTransform.c),
            d: Double(cgTransform.d),
            tx: Double(cgTransform.tx),
            ty: Double(cgTransform.ty))
        let backgroundColor: [Pixel_8] = [0, 200, 200, 200]
        
        let error = withUnsafePointer(to: sourceBuffer) { srcPointer in
            vImageAffineWarpCG_ARGB8888(srcPointer,
                                        &destinationBuffer,
                                        nil,
                                        &vImageTransform,
                                        backgroundColor,
                                        vImage_Flags(kvImageBackgroundColorFill))
        }
        guard error == kvImageNoError else { fatalError("Failed to resize and add padding") }
        
        guard let outputCGImage = try? destinationBuffer.createCGImage(format: sourceFormat ) else {
            print("Error: Failed to create CGImage from vImage_Buffer.")
            return nil
        }
        
        return outputCGImage
    }
    
    func cropAndRotate(cropRect:CGRect, degreeToRotateInClockwise angleInDegrees:Double) -> CGImage? {
        guard let (sourceBuffer, sourceFormat) = getSourceBufferAndFormat() else { return nil }
        defer { sourceBuffer.free() }
        
        let angle = Measurement(value: angleInDegrees,
                                unit: UnitAngle.degrees)
        let radians = CGFloat(angle.converted(to: .radians).value)
        
        let originW = Double(cgImage.width),
            originH = Double(cgImage.height)
        
        // Only allowing 90*n degrees
        if Int(angleInDegrees) % 90 != 0 {
            return nil
        }
        let shouldChangeWidthHeight = Int(angleInDegrees/90) % 2 == 1
        
        
        let cgTransform = CGAffineTransform.identity
            .translatedBy(x: shouldChangeWidthHeight ? cropRect.height/2 : cropRect.width/2, y: shouldChangeWidthHeight ? cropRect.width/2 : cropRect.height/2)
            .rotated(by: -radians)
            .translatedBy(x: -cropRect.width/2, y: -cropRect.height/2)
            .translatedBy(x: -cropRect.minX, y: -(originH - cropRect.maxY))
        
        
        let desiredSize = CGSize(width: shouldChangeWidthHeight ? cropRect.height : cropRect.width,
                                 height: shouldChangeWidthHeight ? cropRect.width : cropRect.height)
        
        guard var destinationBuffer = try? vImage_Buffer(width: Int(desiredSize.width),
                                                         height: Int(desiredSize.height),
                                                         bitsPerPixel: 32) else { return nil }
        defer { destinationBuffer.free() }
        
        var vImageTransform = vImage_CGAffineTransform(
            a: Double(cgTransform.a),
            b: Double(cgTransform.b),
            c: Double(cgTransform.c),
            d: Double(cgTransform.d),
            tx: Double(cgTransform.tx),
            ty: Double(cgTransform.ty))
        let backgroundColor: [Pixel_8] = [0, 200, 200, 200]
        
        let error = withUnsafePointer(to: sourceBuffer) { srcPointer in
            vImageAffineWarpCG_ARGB8888(srcPointer,
                                        &destinationBuffer,
                                        nil,
                                        &vImageTransform,
                                        backgroundColor,
                                        vImage_Flags(kvImageBackgroundColorFill))
        }
        guard error == kvImageNoError else { fatalError("Failed to crop and rotate") }
        
        guard let outputCGImage = try? destinationBuffer.createCGImage(format: sourceFormat ) else {
            print("Error: Failed to create CGImage from vImage_Buffer.")
            return nil
        }
        
        return outputCGImage
    }
}



import UIKit
import CoreGraphics
import CoreML

/// Resizes a UIImage to (1024,1024), normalizes pixel values,
/// reorders channels from HWC → CHW, and returns an MLMultiArray
/// shaped [1, 3, 1024, 1024] (i.e. batch=1, channels=3, height=1024, width=1024).
func prepareImageInputMLMultiArray(originalImage: UIImage) -> MLMultiArray? {
    let targetWidth = 1024
    let targetHeight = 1024
    
    // 1) Resize the image to 1024×1024
    let targetSize = CGSize(width: targetWidth, height: targetHeight)
    UIGraphicsBeginImageContextWithOptions(targetSize, /* opaque */ false, /* scale */ 1.0)
    originalImage.draw(in: CGRect(origin: .zero, size: targetSize))
    guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
        UIGraphicsEndImageContext()
        return nil
    }
    UIGraphicsEndImageContext()
    
    // 2) Get raw RGBA bytes from the resized image
    guard let cgImage = resizedImage.cgImage else {
        return nil
    }
    
    let width = cgImage.width
    let height = cgImage.height
    print("🔥 width, height",width, height)
    let bytesPerPixel = 4
    var rawData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
    
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard
        let context = CGContext(
            data: &rawData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * bytesPerPixel,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )
    else {
        return nil
    }
    
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    
    // 3) Allocate an array to hold our CHW data (3 x height x width)
    //    We'll store floats in [0,1] range.
    var floatCHW = [Float](repeating: 0, count: 3 * width * height)
    

    
    // 4) Normalize each channel and reorder from HWC → CHW
    //    (R, G, B channels in that order).
    for y in 0..<height {
        for x in 0..<width {
            let pixelIndex = (y * width + x) * 4
            let r = Float(rawData[pixelIndex + 0]) / 255.0
            let g = Float(rawData[pixelIndex + 1]) / 255.0
            let b = Float(rawData[pixelIndex + 2]) / 255.0

//case .RRGGBB:
//    let idx = i + j*width
//    res[idx] = r / 255.0
//    res[width*height+idx] = g / 255.0
//    res[width*height*2+idx] = b / 255.0
            // For CHW, index = c*(height*width) + y*width + x
            floatCHW[0 * (height * width) + y * width + x] = r
            floatCHW[1 * (height * width) + y * width + x] = g
            floatCHW[2 * (height * width) + y * width + x] = b
            
//case .RGBRGB:
//    let idx = 3*width*j + i*3
//    res[idx] = r / 255.0
//    res[idx+1] = g / 255.0
//    res[idx+2] = b / 255.0
            
//            let idx = 3*width*y + x*3
//            floatCHW[idx] = r
//            floatCHW[idx+1] = g
//            floatCHW[idx+2] = b
        }
    }
    
    // 5) Create an MLMultiArray of shape [1, 3, height, width] → [1, 3, 1024, 1024]
    //    Then copy our CHW floats into it in row-major order.
    do {
        // shape = [batch=1, channels=3, height=1024, width=1024]
        let shape: [NSNumber] = [1, 3, NSNumber(value: height), NSNumber(value: width)]
        let mlArray = try MLMultiArray(shape: shape, dataType: .float)
        
        // Copy floatCHW → mlArray
        for i in 0..<floatCHW.count {
            mlArray[i] = NSNumber(value: floatCHW[i])
        }
        return mlArray
    } catch {
        print("Error creating MLMultiArray: \(error)")
        return nil
    }
}
