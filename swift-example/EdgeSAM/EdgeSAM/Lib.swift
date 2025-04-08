//
//  Lib.swift
//  EdgeSAM
//
//  Created by seung on 2025/04/03.
//
import Foundation

final class ExecutionMeasurer {
    enum ExecutionKey:String {
        case preprocess, prediction, postprocess, yolox_detection
        case conversionRGB, paddingAndResizing
    }
    
    static private var cache:[String: Double] = [:]
    
    static public func start(_ key: ExecutionKey) {
        cache[key.rawValue] = CFAbsoluteTimeGetCurrent()
    }
    
    @discardableResult
    static public func stop(_ key: ExecutionKey) -> Double {
        guard let start = cache[key.rawValue] else { return 0 }
        let time = CFAbsoluteTimeGetCurrent() - start
        print(String(format: "ExecutionMeasurer [\(key.rawValue)]: %.5f ms.", time*1000))
        cache.removeValue(forKey: key.rawValue)
        return time
    }
}
