using Ply;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

public class RadFoamWebGL : MonoBehaviour
{
    public PlyData Data;
    public float fisheye_fov = 60;
    public Transform Target;


    private Material blitMat;
    private Texture2D positions_tex;
    private Texture2D attr_tex;
    private Texture2D adjacency_tex;
    private Texture2D adjacency_diff_tex;

    private NativeArray<float4> points; // store this for finding the closest cell to the camera on the CPU

    void Start()
    {
        blitMat = new Material(Shader.Find("Hidden/Custom/RadFoamShader"));
        Load();
    }

    void OnDestroy()
    {
        points.Dispose();
        Destroy(blitMat);
    }


    public void Load()
    {
        using var model = Data.Load();

        var vertex_element = model.element_view("vertex");
        var adjacency_element = model.element_view("adjacency");
        var vertex_count = vertex_element.count;
        var adjacency_count = adjacency_element.count;

        var vertex_tex_width = 4096;
        var vertex_tex_height = Mathf.CeilToInt(vertex_count / (float)vertex_tex_width);
        var vertex_tex_size = vertex_tex_width * vertex_tex_height;

        var adj_tex_width = 4096;
        var adj_tex_height = Mathf.CeilToInt(adjacency_count / (float)adj_tex_width);
        var adj_tex_size = adj_tex_width * adj_tex_height;

        // filling buffers one after the other, to ensure we don't run out of memory on webgl
        {
            using var attributes = new NativeArray<half4>(vertex_tex_size, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            new FillColorDataJob {
                r = vertex_element.property_view("red"),
                g = vertex_element.property_view("green"),
                b = vertex_element.property_view("blue"),
                density = vertex_element.property_view("density"),
                attributes = attributes,
            }.Schedule(vertex_count, 512).Complete();
            attr_tex = new Texture2D(vertex_tex_width, vertex_tex_height, TextureFormat.RGBAHalf, 0, true, true);
            attr_tex.filterMode = FilterMode.Point;
            attr_tex.SetPixelData(attributes, 0, 0);
            attr_tex.Apply(false, true);
        }

        points = new NativeArray<float4>(vertex_tex_size, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var points_handle = new FillPointsDataJob {
            x = vertex_element.property_view("x"),
            y = vertex_element.property_view("y"),
            z = vertex_element.property_view("z"),
            adj_offset = vertex_element.property_view("adjacency_offset"),
            points = points
        }.Schedule(vertex_count, 512);

        using var adjacency = new NativeArray<uint>(adj_tex_size, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var adjacency_handle = new ReadUintJob {
            view = adjacency_element.property_view("adjacency"),
            target = adjacency
        }.Schedule(adjacency_count, 512);

        points_handle.Complete();
        adjacency_handle.Complete();
        model.Dispose(); // we can already dispose of the ply model here, as all data is already loaded.. reducing RAM usage maybe?

        using var adj_diff = new NativeArray<half4>(adj_tex_size, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        new BuildAdjDiff { positions = points, adjacency = adjacency, adjacency_diff = adj_diff }.Schedule(vertex_count, 512).Complete();
        adjacency_diff_tex = new Texture2D(adj_tex_width, adj_tex_height, TextureFormat.RGBAHalf, 0, true, true);
        adjacency_diff_tex.filterMode = FilterMode.Point;
        adjacency_diff_tex.SetPixelData(adj_diff, 0, 0);
        adjacency_diff_tex.Apply(false, true);

        positions_tex = new Texture2D(vertex_tex_width, vertex_tex_height, TextureFormat.RGBAFloat, 0, true, true);
        positions_tex.filterMode = FilterMode.Point;
        positions_tex.SetPixelData(points, 0, 0);
        positions_tex.Apply(false, true);

        adjacency_tex = new Texture2D(adj_tex_width, adj_tex_height, TextureFormat.RFloat, 0, true, true);
        adjacency_tex.filterMode = FilterMode.Point;
        adjacency_tex.SetPixelData(adjacency, 0, 0);
        adjacency_tex.Apply(false, true);
    }

    void Update()
    {
        fisheye_fov = Mathf.Clamp(fisheye_fov + Input.mouseScrollDelta.y * -4, 10, 120);

        // if (Input.GetKeyDown(KeyCode.Return)) {
        //     OnDestroy();
        //     using (var model = Model.from_file("C:/Users/Chris/Downloads/scene(3).ply")) {
        //         Load(model, 3);
        //     }
        // }
    }

    void OnRenderImage(RenderTexture srcRenderTex, RenderTexture outRenderTex)
    {
        if (points.Length == 0) {
            Graphics.Blit(srcRenderTex, outRenderTex);
            return;
        }

        var camera = Camera.current;

        var world_to_model = Matrix4x4.Scale(new Vector3(1, -1, 1)) * Target.worldToLocalMatrix;

        {
            blitMat.SetMatrix("_Camera2WorldMatrix", world_to_model * camera.cameraToWorldMatrix);
            blitMat.SetMatrix("_InverseProjectionMatrix", camera.projectionMatrix.inverse);
            blitMat.SetFloat("_FisheyeFOV", fisheye_fov);

            blitMat.SetTexture("_positions_tex", positions_tex);
            blitMat.SetTexture("_adjacency_tex", adjacency_tex);
            blitMat.SetTexture("_adjacency_diff_tex", adjacency_diff_tex);
            blitMat.SetTexture("_attr_tex", attr_tex);

            Graphics.Blit(srcRenderTex, outRenderTex, blitMat);
        }
    }

    [BurstCompile]
    struct FillPointsDataJob : IJobParallelFor
    {
        public PropertyView x;
        public PropertyView y;
        public PropertyView z;
        public PropertyView adj_offset;
        [WriteOnly] public NativeArray<float4> points;

        public void Execute(int index)
        {
            points[index] = new float4(
                x.Get<float>(index),
                y.Get<float>(index),
                z.Get<float>(index),
                adj_offset.Get<float>(index)); // this actually contains a uint, but we just treat it as a float 
        }
    }

    [BurstCompile]
    struct FillColorDataJob : IJobParallelFor
    {
        public int stride;
        public PropertyView r;
        public PropertyView g;
        public PropertyView b;
        public PropertyView density;

        [WriteOnly] public NativeSlice<half4> attributes;

        public void Execute(int index)
        {
            attributes[index] = new half4(
                math.half(r.Get<byte>(index) * (1.0 / (float)byte.MaxValue)),
                math.half(g.Get<byte>(index) * (1.0 / (float)byte.MaxValue)),
                math.half(b.Get<byte>(index) * (1.0 / (float)byte.MaxValue)),
                math.half(density.Get<float>(index))
            );
        }
    }

    [BurstCompile]
    public struct ReadUintJob : IJobParallelFor
    {
        [ReadOnly] public PropertyView view;

        [WriteOnly] public NativeArray<uint> target;

        public void Execute(int index)
        {
            target[index] = view.Get<uint>(index);
        }
    }

    [BurstCompile]
    public struct BuildAdjDiff : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float4> positions;
        [ReadOnly] public NativeArray<uint> adjacency;

        // [NativeDisableParallelForRestriction] may be bad here.. especially as it may run with user provided data, which may be invalid
        [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<half4> adjacency_diff;

        public void Execute(int index)
        {
            float4 cell_data = positions[index];

            int adj_from = (int)(index > 0 ? math.asuint(positions[index - 1].w) : 0);
            int adj_to = (int)math.asuint(cell_data.w);

            for (int a = adj_from; a < adj_to; a++) {
                int adj = (int)adjacency[a];
                float3 adj_pos = positions[adj].xyz;
                float3 adj_diff = adj_pos - cell_data.xyz;

                adjacency_diff[a] = math.half4(new float4(adj_diff, 0));
            }
        }
    }
}
