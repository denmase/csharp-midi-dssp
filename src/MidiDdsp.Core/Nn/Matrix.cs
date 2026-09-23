namespace MidiDdsp.Core.Nn;

/// <summary>
/// A row-major 2-D float array. The networks run on a single example at a
/// time, so a sequence is a matrix of <c>[time, channels]</c> and the batch
/// dimension of the original TensorFlow code is dropped.
/// </summary>
public sealed class Matrix
{
    public Matrix(int rows, int cols)
        : this(rows, cols, new float[CheckedSize(rows, cols)])
    {
    }

    public Matrix(int rows, int cols, float[] data)
    {
        if (rows < 0)
            throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols < 0)
            throw new ArgumentOutOfRangeException(nameof(cols));
        if (data.Length != rows * cols)
            throw new ArgumentException($"Expected {rows * cols} values for a {rows}x{cols} matrix, got {data.Length}.");
        Rows = rows;
        Cols = cols;
        Data = data;
    }

    /// <summary>
    /// Validates <paramref name="rows"/>/<paramref name="cols"/> before the two-arg constructor's
    /// own array allocation, which otherwise runs first (constructor-initializer arguments are
    /// evaluated before either constructor body) -- a negative row/column count used to reach
    /// `new float[...]` with a negative length and fail with OverflowException instead of the
    /// intended ArgumentOutOfRangeException, on every platform (not just this port).
    /// </summary>
    private static int CheckedSize(int rows, int cols)
    {
        if (rows < 0)
            throw new ArgumentOutOfRangeException(nameof(rows));
        if (cols < 0)
            throw new ArgumentOutOfRangeException(nameof(cols));
        return checked(rows * cols);
    }

    public int Rows { get; }
    public int Cols { get; }

    /// <summary>The values in row-major order.</summary>
    public float[] Data { get; }

    public float this[int row, int col]
    {
        get => Data[row * Cols + col];
        set => Data[row * Cols + col] = value;
    }

    public Span<float> Row(int row) => Data.AsSpan(row * Cols, Cols);

    /// <summary>Concatenates matrices with the same number of rows along the columns.</summary>
    public static Matrix ConcatColumns(params Matrix[] parts)
    {
        int rows = parts[0].Rows;
        if (parts.Any(p => p.Rows != rows))
            throw new ArgumentException("All matrices must have the same number of rows.");

        var result = new Matrix(rows, parts.Sum(p => p.Cols));
        for (int r = 0; r < rows; r++)
        {
            var dest = result.Row(r);
            int offset = 0;
            foreach (var part in parts)
            {
                part.Row(r).CopyTo(dest[offset..]);
                offset += part.Cols;
            }
        }
        return result;
    }

    /// <summary>Returns columns <c>[start, start + count)</c> as a new matrix.</summary>
    public Matrix SliceColumns(int start, int count)
    {
        if (start < 0 || count < 0 || start + count > Cols)
            throw new ArgumentOutOfRangeException(nameof(count));
        var result = new Matrix(Rows, count);
        for (int r = 0; r < Rows; r++)
            Row(r).Slice(start, count).CopyTo(result.Row(r));
        return result;
    }
}
