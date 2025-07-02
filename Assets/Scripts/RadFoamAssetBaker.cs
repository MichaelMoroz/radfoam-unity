using Ply;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// Bakes the vertex/ad‑hoc adjacency data into four Texture2D assets and a Material, so the heavy work only
/// happens once in‑editor.  Attach this to a GameObject, assign the *.ply asset and hit the context menu
/// “Bake RadFoam Assets”.
/// </summary>
[ExecuteInEditMode]
public class RadFoamAssetBaker : MonoBehaviour
{
    [Header("Input")]
    public PlyData Data;

    [Header("Settings")]
    [Min(256)] public int textureWidth = 4096;

    [Header("Output (auto‑filled after bake)")]
    public Texture2D positionsTex;
    public Texture2D attrTex;
    public Texture2D adjacencyTex;
    public Texture2D adjacencyDiffTex;
    public Material  bakedMaterial;

    static uint PackFloat3(float3 v)
    {
        float maxv = math.max(math.abs(v.x), math.max(math.abs(v.y), math.abs(v.z)));
        // avoid log2(0) → -∞
        float log2v = maxv > 0f ? math.log2(maxv) : -15f;
        int   e     = (int)math.clamp(math.ceil(log2v), -15f, 15f);

        float scale = math.exp2(-e);
        uint3 sv = (uint3)math.round(math.clamp(v * scale, -1f, 1f) * 255f + 255f);

        return (uint)(e + 15) | (sv.x << 5) | (sv.y << 14) | (sv.z << 23);
    }

#if UNITY_EDITOR
    [ContextMenu("Bake RadFoam Assets")]
    void Bake()
    {
        if (!Data) { Debug.LogWarning("PLY data missing"); return; }

        BuildTextures(Data, textureWidth,
            out positionsTex, out attrTex, out adjacencyTex, out adjacencyDiffTex);

        bakedMaterial = new Material(Shader.Find("Custom/RadFoamShader"))
        {
            name = "RadFoam_Mat"
        };
        bakedMaterial.SetTexture("_positions_tex",        positionsTex);
        bakedMaterial.SetTexture("_adjacency_tex",        adjacencyTex);
        bakedMaterial.SetTexture("_adjacency_diff_tex",   adjacencyDiffTex);
        bakedMaterial.SetTexture("_attr_tex",             attrTex);

        const string folder = "Assets/RadFoamGenerated";
        if (!AssetDatabase.IsValidFolder(folder)) AssetDatabase.CreateFolder("Assets", "RadFoamGenerated");

        SaveAsset(positionsTex,      $"{folder}/positionsTex.asset");
        SaveAsset(attrTex,           $"{folder}/attrTex.asset");
        SaveAsset(adjacencyTex,      $"{folder}/adjacencyTex.asset");
        SaveAsset(adjacencyDiffTex,  $"{folder}/adjacencyDiffTex.asset");
        AssetDatabase.CreateAsset(bakedMaterial, $"{folder}/RadFoam_Mat.mat");

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        Debug.Log("RadFoam bake complete", this);
    }

    static void SaveAsset(Object obj, string path)
    {
        AssetDatabase.CreateAsset(obj, path);
    }
#endif

    static void BuildTextures(PlyData data, int texWidth,
                              out Texture2D positionsTex,
                              out Texture2D attrTex,
                              out Texture2D adjacencyTex,
                              out Texture2D adjacencyDiffTex)
    {
        using var model = data.Load();
        var v = model.element_view("vertex");
        var a = model.element_view("adjacency");

        int vCount = v.count;
        int aCount = a.count;

        int vHeight = Mathf.CeilToInt(vCount / (float)texWidth);
        int vSize   = texWidth * vHeight;

        int aHeight = Mathf.CeilToInt(aCount / (float)texWidth);
        int aSize   = texWidth * aHeight;

        // ───────────────────── vertex colour + density ─────────────────────
        using var attributes = new NativeArray<half4>(vSize, Allocator.TempJob);
        new FillColorDataJob {
            r = v.property_view("red"),
            g = v.property_view("green"),
            b = v.property_view("blue"),
            density = v.property_view("density"),
            attributes = attributes
        }.Schedule(vCount, 512).Complete();

        attrTex = new Texture2D(texWidth, vHeight, TextureFormat.RGBAHalf, 0, true, true) {
            filterMode = FilterMode.Point,
            name       = "RadFoam_Attr"
        };
        attrTex.SetPixelData(attributes, 0, 0);
        attrTex.Apply(false, true);

        // ───────────────────── vertex positions ─────────────────────
        using var points = new NativeArray<float4>(vSize, Allocator.TempJob);
        new FillPointsDataJob {
            x = v.property_view("x"),
            y = v.property_view("y"),
            z = v.property_view("z"),
            adj_offset = v.property_view("adjacency_offset"),
            points = points
        }.Schedule(vCount, 512).Complete();

        positionsTex = new Texture2D(texWidth, vHeight, TextureFormat.RGBAFloat, 0, true, true) {
            filterMode = FilterMode.Point,
            name       = "RadFoam_Pos"
        };
        positionsTex.SetPixelData(points, 0, 0);
        positionsTex.Apply(false, true);

        // ───────────────────── raw adjacency ─────────────────────
        using var adj = new NativeArray<uint>(aSize, Allocator.TempJob);
        new ReadUintJob {
            view   = a.property_view("adjacency"),
            target = adj
        }.Schedule(aCount, 512).Complete();

        adjacencyTex = new Texture2D(texWidth, aHeight, TextureFormat.RFloat, 0, true, true) {
            filterMode = FilterMode.Point,
            name       = "RadFoam_Adj"
        };
        adjacencyTex.SetPixelData(adj, 0, 0);
        adjacencyTex.Apply(false, true);

        // ───────────────────── adjacency-vector diff (packed) ─────────────────────
        using var adjDiff = new NativeArray<float>(aSize, Allocator.TempJob);

        new BuildAdjDiffPacked {
            positions      = points,
            adjacency      = adj,
            adjacency_diff = adjDiff
        }.Schedule(vCount, 512).Complete();

        adjacencyDiffTex = new Texture2D(texWidth, aHeight, TextureFormat.RFloat, 0, true, true) {
            filterMode = FilterMode.Point,
            name       = "RadFoam_AdjDiff"
        };
        adjacencyDiffTex.SetPixelData(adjDiff, 0, 0);
        adjacencyDiffTex.Apply(false, true);
    }

    // ───────────────────────────────────────────────────────────── jobs ─────────────────────────────────────────────────────────────

    [BurstCompile]
    struct FillPointsDataJob : IJobParallelFor
    {
        public PropertyView x, y, z, adj_offset;
        [WriteOnly] public NativeArray<float4> points;

        public void Execute(int i)
        {
            points[i] = new float4(
                x.Get<float>(i),
                y.Get<float>(i),
                z.Get<float>(i),
                adj_offset.Get<float>(i)); // offset fits safely in fp32.
        }
    }

    [BurstCompile]
    struct FillColorDataJob : IJobParallelFor
    {
        public PropertyView r, g, b, density;
        [WriteOnly] public NativeSlice<half4> attributes;

        public void Execute(int i)
        {
            const float inv = 1f / 255f;
            attributes[i] = new half4(
                math.half(r.Get<byte>(i) * inv),
                math.half(g.Get<byte>(i) * inv),
                math.half(b.Get<byte>(i) * inv),
                math.half(density.Get<float>(i)));
        }
    }

    [BurstCompile]
    struct ReadUintJob : IJobParallelFor
    {
        [ReadOnly]  public PropertyView view;
        [WriteOnly] public NativeArray<uint> target;
        public void Execute(int i) => target[i] = view.Get<uint>(i);
    }

    [BurstCompile]
    struct BuildAdjDiffPacked : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float4> positions;   // xyz = pos, w = adj offset (u32 bits)
        [ReadOnly] public NativeArray<uint>   adjacency;   // compact indices
        [WriteOnly, NativeDisableParallelForRestriction]
        public NativeArray<float>             adjacency_diff; // R32F expects float

        public void Execute(int i)
        {
            float3 p = positions[i].xyz;

            int first = i > 0 ? (int)math.asuint(positions[i - 1].w) : 0;
            int last  =          (int)math.asuint(positions[i].w);

            for (int a = first; a < last; a++)
            {
                float3 diff   = positions[(int)adjacency[a]].xyz - p;
                uint   packed = PackFloat3(diff);
                adjacency_diff[a] = math.asfloat(packed);   // bit-cast
            }
        }
    }
}
