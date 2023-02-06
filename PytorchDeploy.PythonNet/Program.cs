using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using HalconDotNet;
using Python.Runtime;
using PytorchDeploy.ClassLibrary;
using static HalconDotNet.HOperatorSet;

var logFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "log.txt");
if (File.Exists(logFilePath))
    File.Delete(logFilePath);
var resultImageDir = Path.Join(AppDomain.CurrentDomain.BaseDirectory, "result");
const string imagePathDir = @"F:\Library\Desktop\FTT\SpliteImages\Source";
var imagePaths = Directory.GetFiles(imagePathDir, "*.png");
const int batchSize = 6;
var imageChunk = imagePaths.Chunk(batchSize);
//dir is exist,delete it,than create it
if (Directory.Exists(resultImageDir))
    Directory.Delete(resultImageDir, true);
Directory.CreateDirectory(resultImageDir);
Runtime.PythonDLL = @"C:\Users\LolitaComplex\.conda\envs\CudaDeepLab\python39.dll";
PythonEngine.PythonHome = @"C:\Users\LolitaComplex\.conda\envs\CudaDeepLab";
var pythonPathBuilder = new StringBuilder();
pythonPathBuilder.Append(@"C:\Users\LolitaComplex\AppData\Roaming\Python\Python39\site-packages;");
pythonPathBuilder.Append(@"C:\Users\LolitaComplex\.conda\envs\CudaDeepLab\Lib\site-packages;");
pythonPathBuilder.Append(@"C:\Users\LolitaComplex\.conda\envs\CudaDeepLab\DLLs;");
pythonPathBuilder.Append(@"C:\Users\LolitaComplex\.conda\envs\CudaDeepLab\Lib;");
pythonPathBuilder.Append(@"F:\Library\Documents\Source\Repo\CudaDeepLab;");
PythonEngine.PythonPath = pythonPathBuilder.ToString();
PythonEngine.Initialize();
var isInitialized = PythonEngine.IsInitialized;
$"PythonEngine.IsInitialized: {isInitialized}".WriteLine(isInitialized ? ConsoleColor.Green : ConsoleColor.Red);
using (Py.GIL())
{
    using (var scope = Py.CreateScope())
    {
        dynamic pyObject = scope.Import("csharp_run");
        var model = pyObject.init_model(@"F:\Library\Documents\Source\Repo\CudaDeepLab\logs\best_epoch_weights.pth");
        foreach (var batchs in imageChunk)
        {
            var stringBuilder = new StringBuilder();
            string log = default;
            var buffHTuple = new HTuple();
            var sourceImages = new List<(HObject image, string imagePath)>(batchSize);
            foreach (var imagePath in batchs)
            {
                ReadImage(out var image, imagePath);
                sourceImages.Add((image, imagePath));
                var t1 = Stopwatch.GetTimestamp();
                Decompose3(image, out var rImage, out var gImage, out var bImage);
                GetRegionPoints(rImage, out var rows, out var cols);
                GetGrayval(rImage, rows, cols, out var rValues);
                GetGrayval(gImage, rows, cols, out var gValues);
                GetGrayval(bImage, rows, cols, out var bValues);
                buffHTuple = buffHTuple.TupleConcat(rValues, gValues, bValues);
                var t2 = Stopwatch.GetTimestamp();
                log = $"图像转化{new TimeSpan(t2 - t1).TotalMilliseconds}ms";
                log.WriteLine();
                stringBuilder.AppendLine(log);
                rImage.Dispose();
                gImage.Dispose();
                bImage.Dispose();
            }

            var t3 = Stopwatch.GetTimestamp();
            var arr = buffHTuple.ToIArr();
            var t4 = Stopwatch.GetTimestamp();
            log = $"HTupleToArr图像转化{new TimeSpan(t4 - t3).TotalMilliseconds}ms";
            log.WriteLine();
            stringBuilder.AppendLine(log);
            var t5 = Stopwatch.GetTimestamp();
            var outputPtrs = Enumerable.Range(0, 6)
                .Select(i =>
                {
                    var buffer = Marshal.AllocHGlobal(sizeof(byte) * 512 * 512);
                    return buffer.ToInt64();
                })
                .ToArray();
            unsafe
            {
                fixed (int* pArr = arr)
                {
                    var nintr = (long)pArr;
                    pyObject.eval(model, nintr, outputPtrs);
                }
            }

            var t6 = Stopwatch.GetTimestamp();
            log = $"Pytorch推理{new TimeSpan(t6 - t5).TotalMilliseconds}ms";
            log.WriteLine();
            stringBuilder.AppendLine(log);
            var packages = new List<(HObject image, string imagePath, nint imagePtr)>(batchs.Length);
            for (var i = 0; i < batchs.Length; i++)
            {
                var r = outputPtrs[i];
                var (image, imagePath) = sourceImages[i];
                packages.Add((image, imagePath, (nint)r));
            }

            foreach (var (image, imagePath, imagePtr) in packages)
            {
                var t7 = Stopwatch.GetTimestamp();
                byte[] array;
                unsafe
                {
                    var span = new Span<byte>((void*)imagePtr, 512 * 512);
                    array = span.ToArray();
                }

                GenImage1(out var segImage, "byte", 512, 512, imagePtr);
                Threshold(segImage, out var region1, 1, 1);
                Threshold(segImage, out var region2, 2, 2);
                OverpaintRegion(image, region1, new[] { 255, 0, 0 }, "margin");
                OverpaintRegion(image, region2, new[] { 0, 255, 0 }, "margin");
                var t8 = Stopwatch.GetTimestamp();
                log = $"后处理{new TimeSpan(t8 - t7).TotalMilliseconds}ms";
                log.WriteLine();
                stringBuilder.AppendLine(log);
                WriteImage(image, "tiff", 0,
                    Path.Join(resultImageDir, $"{Path.GetFileNameWithoutExtension(imagePath)}.tiff"));
                segImage.Dispose();
                region1.Dispose();
                region2.Dispose();
                image.Dispose();
                Marshal.FreeHGlobal(imagePtr);
            }

            log = "-----------------------";
            log.WriteLine();
            stringBuilder.AppendLine(log);
            File.AppendAllLines(logFilePath, stringBuilder.ToString().Split(Environment.NewLine));
        }
    }
}

PythonEngine.Shutdown();
Environment.Exit(0);