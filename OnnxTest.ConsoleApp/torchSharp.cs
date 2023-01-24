// using System;
// using System.Collections.Generic;
// using System.Diagnostics;
// using System.IO;
// using System.Linq;
// using System.Text;
// using HalconDotNet;
// using TorchSharp;
// using static HalconDotNet.HOperatorSet;
//
// unsafe
// {
//     var logFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "log.txt");
//     if (File.Exists(logFilePath))
//         File.Delete(logFilePath);
//     //set env variable export PYTORCH_NVFUSER_DISABLE=fallback
//     const string envVariableStr = "PYTORCH_NVFUSER_DISABLE";
//     Environment.SetEnvironmentVariable(envVariableStr, "fallback");
//     var resultImageDir = Path.Join(AppDomain.CurrentDomain.BaseDirectory, "result");
//     const string imagePathDir = @"F:\Library\Desktop\FTT\SpliteImages\Source";
//     var imagePaths = Directory.GetFiles(imagePathDir, "*.png");
//     const int batchSize = 6;
//     var imageChunk = imagePaths.Chunk(batchSize);
//     //dir is exist,delete it,than create it
//     if (Directory.Exists(resultImageDir))
//         Directory.Delete(resultImageDir, true);
//     Directory.CreateDirectory(resultImageDir);
//     const string modelPath = @"F:\Library\Documents\Source\Repo\CudaDeepLab\model_data\deeplabv3_plus_mobilenet.pt";
//     var module = torch.jit.load(modelPath, DeviceType.CUDA);
//     module.eval();
//     foreach (var batchs in imageChunk)
//     {
//         var stringBuilder = new StringBuilder();
//         string log = default;
//         var buffHTuple = new HTuple();
//         var sourceImages = new List<(HObject image, string imagePath)>(batchSize);
//         foreach (var imagePath in batchs)
//         {
//             ReadImage(out var image, imagePath);
//             sourceImages.Add((image, imagePath));
//             var t1 = Stopwatch.GetTimestamp();
//             Decompose3(image, out var rImage, out var gImage, out var bImage);
//             GetRegionPoints(rImage, out var rows, out var cols);
//             GetGrayval(rImage, rows, cols, out var rValues);
//             GetGrayval(gImage, rows, cols, out var gValues);
//             GetGrayval(bImage, rows, cols, out var bValues);
//             buffHTuple = buffHTuple.TupleConcat(rValues, gValues, bValues);
//             var t2 = Stopwatch.GetTimestamp();
//             log = $"图像转化{new TimeSpan(t2 - t1).TotalMilliseconds}ms";
//             log.WriteLine();
//             stringBuilder.AppendLine(log);
//             rImage.Dispose();
//             gImage.Dispose();
//             bImage.Dispose();
//         }
//
//         buffHTuple /= 255.0;
//         var t3 = Stopwatch.GetTimestamp();
//         var input = torch.tensor(buffHTuple.ToDArr(), new long[] { batchs.Length, 3, 512, 512 },
//             torch.ScalarType.Float32, torch.device(DeviceType.CUDA));
//         var t4 = Stopwatch.GetTimestamp();
//         log = $"Tensor创建{new TimeSpan(t4 - t3).TotalMilliseconds}ms";
//         log.WriteLine();
//         stringBuilder.AppendLine(log);
//         var t5 = Stopwatch.GetTimestamp();
//         using var grad = torch.no_grad();
//         var results = ((torch.Tensor)module.forward(input)).cpu();
//         var t6 = Stopwatch.GetTimestamp();
//         log = $"推理{new TimeSpan(t6 - t5).TotalMilliseconds}ms";
//         log.WriteLine();
//         stringBuilder.AppendLine(log);
//         var packages = new List<(HObject image, string imagePath, torch.Tensor tensor)>(batchs.Length);
//         for (var i = 0; i < batchs.Length; i++)
//         {
//             var result = results[i];
//             var (image, imagePath) = sourceImages[i];
//             packages.Add((image, imagePath, result));
//         }
//
//         foreach (var (image, imagePath, tensor) in packages)
//         {
//             var t7 = Stopwatch.GetTimestamp();
//             var permute = tensor.permute(1, 2, 0);
//             var softmax = torch.nn.functional.softmax(permute, -1);
//             var argmax = softmax.argmax(-1);
//             var tensorAccessor = argmax.data<long>();
//             var array = tensorAccessor.Select(l => (byte)l).ToArray();
//             fixed (byte* l = array)
//             {
//                 GenImage1(out var segImage, "byte", 512, 512, (nint)l);
//                 Threshold(segImage, out var region1, 1, 1);
//                 Threshold(segImage, out var region2, 2, 2);
//                 OverpaintRegion(image, region1, new[] { 255, 0, 0 }, "margin");
//                 OverpaintRegion(image, region2, new[] { 0, 255, 0 }, "margin");
//                 var t8 = Stopwatch.GetTimestamp();
//                 log = $"后处理{new TimeSpan(t8 - t7).TotalMilliseconds}ms";
//                 log.WriteLine();
//                 stringBuilder.AppendLine(log);
//                 WriteImage(image, "tiff", 0,
//                     Path.Join(resultImageDir, $"{Path.GetFileNameWithoutExtension(imagePath)}.tiff"));
//                 segImage.Dispose();
//                 region1.Dispose();
//                 region2.Dispose();
//             }
//
//             tensor.Dispose();
//             permute.Dispose();
//             softmax.Dispose();
//             argmax.Dispose();
//             tensorAccessor.Dispose();
//             image.Dispose();
//         }
//
//         log = "-----------------------";
//         log.WriteLine();
//         stringBuilder.AppendLine(log);
//         File.AppendAllLines(logFilePath, stringBuilder.ToString().Split(Environment.NewLine));
//     }
//
//     Environment.SetEnvironmentVariable(envVariableStr, null);
//     Process.Start("explorer.exe", resultImageDir);
// }
//
