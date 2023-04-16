using Microsoft.OpenApi.Models;
using Microsoft.IdentityModel.Logging;
using Microsoft.Identity.Web;
using Microsoft.AspNetCore.Authorization;

namespace WebGrip.Putty.Brazen
{
    public class Startup
    {
        public IConfiguration Configuration { get; }

        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public void ConfigureServices(IServiceCollection services)
        {
            services.AddOptions();

            //var instance = Configuration["AzureAd:Instance"];
            //var tenantId = Configuration["AzureAd:TenantId"];
            //var clientId = Configuration["AzureAd:ClientId"];
            //var clientSecret = Configuration["AzureAd:ClientSecret"];

            services.Configure<CookiePolicyOptions>(options => // TODO look into this more, what can we put in here?
            {
                // This lambda determines whether user consent for non-essential cookies is needed for a given request.
                options.CheckConsentNeeded = context => true;
                options.MinimumSameSitePolicy = SameSiteMode.Unspecified;
                // Handling SameSite cookie according to https://docs.microsoft.com/en-us/aspnet/core/security/samesite?view=aspnetcore-3.1
                options.HandleSameSiteCookieCompatibility();
            });


            ConfigureSwagger(services);
        }


        private void ConfigureSwagger(IServiceCollection services)
        {
            Dictionary<string, string> scopes = new Dictionary<string, string> // TODO this needs to go into appsettings.json
            {
                //{ "https://graph.microsoft.com/.default", "Graph" },
                { "User.Read", "Reading user" }
            };


            services.AddGrpc().AddJsonTranscoding();
            services.AddGrpcSwagger();

            services.AddSwaggerGen(c =>
            {
                c.SwaggerDoc(
                    "v1",
                    new OpenApiInfo
                    {
                        Title = "gRPC transcoding",
                        Version = "v1"
                    }
                );
            });
        }

        public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
        {
            if (env.IsDevelopment())
            {
                IdentityModelEventSource.ShowPII = true;
                IdentityModelEventSource.HeaderWritten = true;
                app.UseDeveloperExceptionPage();
            }

            app.UseHttpsRedirection();
            app.UseRouting();

            app.UseAuthentication();
            app.UseAuthorization();

            app.UseSwagger();

            app.UseSwaggerUI(c => {
                c.OAuthClientId(Configuration["AzureAd:ClientId"]);
                c.SwaggerEndpoint("/swagger/v1/swagger.json", "My API V1");
            });


            app.UseEndpoints(endpoints =>
            {
                endpoints.MapGrpcService<ChatService>();
                endpoints.MapGet("/", async context =>
                {
                    await context.Response.WriteAsync("Communication with gRPC endpoints must be made through a gRPC client. To learn how to create a client, visit: https://go.microsoft.com/fwlink/?linkid=2086909");
                });
            });
        }
        public class HasScopeRequirement : IAuthorizationRequirement
        {
            public string Scope { get; }
            public string Issuer { get; }

            public HasScopeRequirement(string scope, string issuer)
            {
                Scope = scope;
                Issuer = issuer;
            }
        }
    }
}
