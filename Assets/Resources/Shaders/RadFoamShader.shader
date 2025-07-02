Shader "Custom/RadFoamShader"
{
    Properties
    {
        _attr_tex ("Attributes Texture", 2D) = "white" {}
        _positions_tex ("Positions Texture", 2D) = "white" {}
        _adjacency_diff_tex ("Adjacency Diff Texture", 2D) = "white" {}
        _adjacency_tex ("Adjacency Texture", 2D) = "white" {}

        _TransmittanceThreshold ("Transmittance Threshold", Range(0, 1)) = 0.1

        _BBoxPos ("Bounding Box Position", Vector) = (0, 0, 0, 0)
        _BBoxSize ("Bounding Box Size", Vector) = (15, 15, 15, 0)
        _BBoxRotation ("Bounding Box Rotation", Vector) = (0, 0, 0, 1)
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

            Texture2D<float> _adjacency_diff_tex; 
            Texture2D<float> _adjacency_tex;

            float _TransmittanceThreshold;
            float4 _BBoxPos;
            float4 _BBoxSize;
            float4 _BBoxRotation;
            
            #define WIDTH_BITS 13
            #define WIDTH 8192

            #define MID_ESP 0.005f

            float3 unpackfloat3(uint data)
            {
                float scale = exp2(float(data & 0x1Fu) - 23.0);
                float3 sv = float3((data >> 5) & 0x1FFu, (data >> 14) & 0x1FFu, data >> 23);
                float offset = 255.0 * scale;
                return scale * sv - offset;
            }

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

            uint adjacency_diff_buffer(uint i) {
                return asuint(_adjacency_diff_tex[index_to_tex_buffer(i)]);
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
            uint FindNearestCell(float3 pos, uint initial_guess)
            {
                uint cell = initial_guess;
                float closest_distance = 1e10;
                for (uint i = 0; i < 1024; i++) {
                    float4 cell_data = positions_buff(cell);
                    float3 cell_pos = cell_data.xyz;

                    uint next_face = -1; 
                    uint adj_from = cell > 0 ? asuint(positions_buff(cell - 1).w) : 0;
                    uint adj_to = asuint(cell_data.w);
                    for (uint f = adj_from; f < adj_to; f++) {
                        float3 diff = unpackfloat3(adjacency_diff_buffer(f));
                        float3 adj_pos = cell_pos + diff;
                        float dist = distance(pos, adj_pos);
                        if(dist < closest_distance)
                        {
                            next_face = f;
                            closest_distance = dist;
                        }
                    }

                    if (next_face == -1) {
                        break;
                    }

                    cell = adjacency_buffer(next_face);
                }
                return cell;
            }

            // rotate v by unit quaternion q
            float3 qRotate(float3 v, float4 q)
            {
                float3 t = 2.0 * cross(q.xyz, v);
                return v + q.w * t + cross(q.xyz, t);
            }

            float4 eulerToQuat(float3 euler)
            {
                float c1 = cos(euler.x * 0.5);
                float c2 = cos(euler.y * 0.5);
                float c3 = cos(euler.z * 0.5);
                float s1 = sin(euler.x * 0.5);
                float s2 = sin(euler.y * 0.5);
                float s3 = sin(euler.z * 0.5);

                return float4(s1 * c2 * c3 - c1 * s2 * s3,
                              c1 * s2 * c3 + s1 * c2 * s3,
                              c1 * c2 * s3 - s1 * s2 * c3,
                              c1 * c2 * c3 + s1 * s2 * s3);
            }

            // ray–OBB intersection:  (-1,-1)  if miss
            float2 IntersectRayBox(
                float3 rayOrigin, float3 rayDir,
                float3 boxPos,   float3 boxScale,   float4 boxQuat)
            {
                // transform ray into box space  (conjugate == inverse for unit quats)
                float4 qc      = float4(-boxQuat.xyz, boxQuat.w);
                float3 oLocal  = qRotate(rayOrigin - boxPos, qc);
                float3 dLocal  = qRotate(rayDir,           qc);

                // slab test in box space, box extents = ±boxScale
                float3 invD = 1.0 / dLocal;
                float3 n    = invD * oLocal;
                float3 k    = abs(invD) * boxScale;

                float3 t1 = -n - k;
                float3 t2 = -n + k;

                float tNear = max(max(t1.x, t1.y), t1.z);
                float tFar  = min(min(t2.x, t2.y), t2.z);

                return (tNear > tFar || tFar < 0.0) ? float2(-1.0, -1.0)
                                                    : float2(tNear, tFar);
            }

            v2f vert (appdata v)
            {
                v2f o;
                UNITY_SETUP_INSTANCE_ID(v);
                UNITY_INITIALIZE_OUTPUT(v2f, o);
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
                o.pos = float4(float2(1,-1)*(v.uv*2-1),1.0,1);
                o.start = FindNearestCell(mul(unity_WorldToObject, float4(_WorldSpaceCameraPos, 1)).xyz, 123);
                return o;
            }

            #define CHUNK_SIZE 4
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
                
                Ray ray;
                ray.origin = rayPos;
                ray.direction = rayDir;

                float t_0 = 0.0f;
                float t_max = dot(worldPos - rayPos, rayDir);

                // Calculate the intersection with the bounding box
                float2 t = IntersectRayBox(rayPos, rayDir, _BBoxPos.xyz, _BBoxSize.xyz, eulerToQuat(_BBoxRotation.xyz));

                t_0 = max(t.x, t_0);
                t_max = min(t_max, t.y);

                if (t.y < 0 || t_max <= t_0) {
                    return float4(0, 0, 0, 0); // No intersection with the volume
                }

                uint enc_diffs[CHUNK_SIZE];

                // tracing state
                uint cell = input.start;

                if(t.x > 0) { // Find new starting point if outside the bounding box
                    float3 start_pos = ray.origin + ray.direction * t_0;
                    cell = FindNearestCell(start_pos, cell);
                }

                float transmittance = 1.0f;
                float3 color = float3(0, 0, 0);

                int i = 0;
                for (; i < 192; i++) {
                    float4 cell_data = positions_buff(cell);
                    uint adj_from = cell > 0 ? asuint(positions_buff(cell - 1).w) : 0;
                    uint adj_to = asuint(cell_data.w);

                    float t_1 = 1e10f;
                    uint next_face = -1; 

                    float3 cell_offset = cell_data.xyz - ray.origin;
                    for (uint f = adj_from; f < adj_to; f++) {
                        float3 diff = unpackfloat3(adjacency_diff_buffer(f));
                        float denom = dot(diff, ray.direction);
                        float3 mid = diff * (0.5 + MID_ESP) + cell_offset;
                        float t = dot(mid, diff) / denom;
                        if(denom > 0 && t < t_1) {
                            t_1 = t;
                            next_face = f;
                        }
                    }

                    t_1 = min(t_1, t_max);
        
                    float4 attrs = attrs_buff(cell);
                    float density = attrs.w;
                    float alpha = 1.0 - exp(-density * (t_1 - t_0));
                    float weight = transmittance * alpha;

                    color += attrs.rgb * weight;
                    transmittance = transmittance * (1.0 - alpha);

                    if (next_face == -1 || t_1 >= t_max || transmittance < _TransmittanceThreshold) {
                        break;
                    }

                    cell = adjacency_buffer(next_face);
                    t_0 = t_1;
                }

                //color = i / 200.0;
                //transmittance = 0.0;
                color = pow(color, 2.2f); // Fix color
                if (transmittance <= _TransmittanceThreshold) transmittance = 0.0;
                return float4(color, 1.0f - transmittance);
            }
            ENDCG
        }
    }
}
