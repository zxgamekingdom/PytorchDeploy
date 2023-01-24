using System;

public static class ConsoleExtensions
{
    public static void WriteLine<T>(this T t, ConsoleColor? backgroundColor = default,
        ConsoleColor? foregroundColor = default)
    {
        var writer = Console.Out;
        lock (writer)
        {
            var defaultBackgroundColor = Console.BackgroundColor;
            var defaultForegroundColor = Console.ForegroundColor;
            Console.BackgroundColor = backgroundColor ?? defaultBackgroundColor;
            Console.ForegroundColor = foregroundColor ?? defaultForegroundColor;
            Console.WriteLine(t);
            Console.BackgroundColor = defaultBackgroundColor;
            Console.ForegroundColor = defaultForegroundColor;
        }
    }

    public static string ArrayStr<T>(this T[] array)
    {
        return string.Join(',', array);
    }
}