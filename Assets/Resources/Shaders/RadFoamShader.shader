Shader "Custom/RadFoamShader"
{
    Properties
    {
        _attr_tex ("Attributes Texture", 2D) = "white" {}
        _positions_tex ("Positions Texture", 2D) = "white" {}
        _adjacency_diff_tex ("Adjacency Diff Texture", 2D) = "white" {}
        _adjacency_tex ("Adjacency Texture", 2D) = "white" {}
    }
    SubShader
	{
		Tags { "Queue" = "Overlay" "LightMode" = "Always" "LightMode" = "ForwardBase" }
		Pass
		{
            ZWrite Off
            ZTest Always
            Cull Off
            Blend One OneMinusSrcAlpha

            CGPROGRAM
            #pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"
            #include "UnityLightingCommon.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            #if defined(UNITY_VERTEX_INPUT_INSTANCE_ID)
                UNITY_VERTEX_INPUT_INSTANCE_ID
            #endif
            };       

            struct v2f
			{
				float4 pos : SV_POSITION;
                uint start : TEXCOORD0;
                UNITY_VERTEX_OUTPUT_STEREO
			};

            struct Ray
            {
                float3 origin;
                float3 direction;
            };

            UNITY_DECLARE_DEPTH_TEXTURE(_CameraDepthTexture);

            Texture2D<float4> _attr_tex;
            Texture2D<float4> _positions_tex;

            Texture2D<float4> _adjacency_diff_tex; 
            Texture2D<float> _adjacency_tex;
            
            #define WIDTH_BITS 12
            #define WIDTH 4096

            uint2 index_to_tex_buffer(uint i) {
                return uint2(i & (WIDTH - 1), i >> WIDTH_BITS);
            }

            float4 positions_buff(uint i) {
                return _positions_tex[index_to_tex_buffer(i)];
            }

            float4 attrs_buff(uint i) {
                return _attr_tex[index_to_tex_buffer(i)];
            }

            uint adjacency_buffer(uint i) {
                return asuint(_adjacency_tex[index_to_tex_buffer(i)]);
            }

            float3 adjacency_diff_buffer(uint i) {
                return _adjacency_diff_tex[index_to_tex_buffer(i)].xyz;
            }

            float4 SVPositionToClipPos(float4 pos)
            {
                float4 clipPos = float4(((pos.xy / _ScreenParams.xy) * 2 - 1) * int2(1, -1), pos.z, 1);
                #ifdef UNITY_SINGLE_PASS_STEREO
                    clipPos.x -= 2 * unity_StereoEyeIndex;
                #endif
                return clipPos;
            }

            // thanvolumeDensity lyuma & cnlohr for the base of this.
            // it needed some modifications accounting for flipped projection & mirrors
            float4 ClipToViewPos(float4 clipPos)
            {
                float4 normalizedClipPos = float4(clipPos.xyz / clipPos.w, 1);
                normalizedClipPos.z = 1 - normalizedClipPos.z;
                normalizedClipPos.z = normalizedClipPos.z * 2 - 1;
                float4x4 invP = unity_CameraInvProjection;
                // do projection flip on this, found empirically
                invP._24 *= _ProjectionParams.x;
                // this is needed for mirrors to work properly, found empirically
                invP._42 *= -1;
                float4 viewPos = mul(invP, normalizedClipPos);
                // and the y coord needs to flip for flipped projection, found empirically
                viewPos.y *= _ProjectionParams.x;
                return viewPos;
            }

            // the same as the previous function, but it precalculates all the operations as one matrix
            float4x4 CreateClipToViewMatrix()
            {
                float4x4 flipZ = float4x4(1, 0, 0, 0,
                                          0, 1, 0, 0,
                                          0, 0, -1, 1,
                                          0, 0, 0, 1);
                float4x4 scaleZ = float4x4(1, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 2, -1,
                                           0, 0, 0, 1);
                float4x4 invP = unity_CameraInvProjection;
                float4x4 flipY = float4x4(1, 0, 0, 0,
                                          0, _ProjectionParams.x, 0, 0,
                                          0, 0, 1, 0,
                                          0, 0, 0, 1);

                float4x4 result = mul(scaleZ, flipZ);
                result = mul(invP, result);
                result = mul(flipY, result);
                result._24 *= _ProjectionParams.x;
                result._42 *= -1;
                return result;
            }

            //Traverse the adjacency graph to find the nearest cell (finds nearest cell to given position in N^(1/3) time, where N is the number of cells)
            uint FindNearestCell(float3 pos) 
            {
                uint cell = 100;
                float closest_distance = 1e10;
                for (uint i = 0; i < 1024; i++) {
                    float4 cell_data = positions_buff(cell);
                    float3 cell_pos = cell_data.xyz;

                    uint next_face = 0xFFFFFFFF; 
                    uint adj_from = cell > 0 ? asuint(positions_buff(cell - 1).w) : 0;
                    uint adj_to = asuint(cell_data.w);
                    for (uint f = adj_from; f < adj_to; f++) {
                        half3 diff = adjacency_diff_buffer(f).xyz;
                        float3 adj_pos = cell_pos + diff;
                        float dist = distance(pos, adj_pos);
                        if(dist < closest_distance)
                        {
                            next_face = f;
                            closest_distance = dist;
                        }
                    }

                    if (next_face == 0xFFFFFFFF) {
                        break;
                    }

                    cell = adjacency_buffer(next_face);
                }
                return cell;
            }

            v2f vert (appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.pos = float4(float2(1,-1)*(v.uv*2-1),1.0,1);
                o.start = FindNearestCell(mul(unity_WorldToObject, float4(_WorldSpaceCameraPos, 1)).xyz);
                return o;
            }

            #define CHUNK_SIZE 5
            float4 frag(v2f input) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);

                // get the clip position and sample the depth texture
                float4 clipPos = SVPositionToClipPos(input.pos);
                float4 uv = ComputeScreenPos(clipPos);
                float depth = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, uv.xy / uv.w);

                float4x4 invP = CreateClipToViewMatrix();
                // construct the clip space position from SV_Position & the depth texture
                // then simply multiply it by the inverse projection matrix
                float4 viewPos = mul(invP, float4(clipPos.xy / clipPos.w, depth, 1));
                // don't forget to do the perspective divide
                viewPos = float4(viewPos.xyz / viewPos.w, 1);
                // there you go, world space position is just one more matrix multiplication away
                float3 worldPos = mul(UNITY_MATRIX_I_V, viewPos).xyz;

                float3 rayPos = _WorldSpaceCameraPos;
                float3 rayDir = worldPos - rayPos;

                // Transform to model space
                rayPos = mul(unity_WorldToObject, float4(rayPos, 1)).xyz;
                rayDir = normalize(mul(unity_WorldToObject, float4(rayDir, 0)).xyz);
                worldPos = mul(unity_WorldToObject, float4(worldPos, 1)).xyz;

                float max_t = dot(worldPos - rayPos, rayDir);
           
                Ray ray;
                ray.origin = rayPos;
                ray.direction = rayDir;

                float scene_depth = 10000; 
                float3 diffs[CHUNK_SIZE];

                // tracing state
                uint cell = input.start;
                float transmittance = 1.0f;
                float3 color = float3(0, 0, 0);
                float t_0 = 0.0f;

                int i = 0;
                for (; i < 200 && transmittance > 0.05; i++) {
                    float4 cell_data = positions_buff(cell);
                    uint adj_from = cell > 0 ? asuint(positions_buff(cell - 1).w) : 0;
                    uint adj_to = asuint(cell_data.w);

                    float4 attrs = attrs_buff(cell);

                    float t_1 = scene_depth;
                    uint next_face = 0xFFFFFFFF; 

                    uint faces = adj_to - adj_from;
                    for (uint f = 0; f < faces; f += CHUNK_SIZE) {

                        [unroll(CHUNK_SIZE)]
                        for (uint a1 = 0; a1 < CHUNK_SIZE; a1++) {
                            diffs[a1] = adjacency_diff_buffer(adj_from + f + a1).xyz;
                        }

                        [unroll(CHUNK_SIZE)]
                        for (uint a2 = 0; a2 < CHUNK_SIZE; a2++) {
                            half3 diff = diffs[a2];
                            float denom = dot(diff, ray.direction);
                            float3 mid = cell_data.xyz + diff * 0.5f;
                            float t = dot(mid - ray.origin, diff) / denom;
                            bool valid = denom > 0 && t < t_1 && t > t_0 && f + a2 < faces;
                            t_1 = valid ? t : t_1;
                            next_face = valid ? adj_from + f + a2 : next_face;
                        }
                    }

                    t_1 = min(t_1, max_t);

                    float density = attrs.w;
                    float alpha = 1.0 - exp(-density * (t_1 - t_0));
                    float weight = transmittance * alpha;

                    color += attrs.rgb * weight;
                    transmittance = transmittance * (1.0 - alpha);

                    if (next_face == 0xFFFFFFFF || t_1 >= max_t) {
                        break;
                    }

                    cell = adjacency_buffer(next_face);
                    t_0 = t_1;
                }

                color = pow(color, 2.2f); // Fix color
                return float4(color, 1.0f - transmittance);
            }
            ENDCG
        }
    }
}
